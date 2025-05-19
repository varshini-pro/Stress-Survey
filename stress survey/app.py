
from flask import Flask, request, jsonify, render_template
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load the model, scaler, and column names
model = load_model('stress_prediction_model.h5')
scaler = joblib.load('scaler.pkl')
X_columns = joblib.load('X_columns.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    data = request.form.to_dict()
    df = pd.DataFrame([data])

    # Ensure numerical values are interpreted correctly
    df = df.apply(pd.to_numeric, errors='ignore')

    # Process 'Age' column if it exists
    if 'Age' in df.columns:
        age_mapping = {
            '18-20': 19,
            '21-23': 22,
            '24-26': 25,
            '> 25': 26
        }
        df['Age'] = df['Age'].map(age_mapping)
    
    # Handle other necessary preprocessing steps here
    categorical_columns = ["The name of your institution", "The name of your program of study", "Your current class level is", 
                           "Your gender", 'Living with family?', 'Are you happy with your academic Condition?', 
                           'Are you addicted to any drugs?', 'Are you in a relationship?']
    # Strip any spaces in column names
    categorical_columns = [col.strip() for col in categorical_columns]
    
    # Check for missing columns and add them with default values
    for col in categorical_columns:
        if col not in df.columns:
            df[col] = None  # or some default value

    # Ensure all columns are properly encoded
    df = pd.get_dummies(df, columns=categorical_columns)
    
    # Ensure all columns match training columns
    df = df.reindex(columns=X_columns, fill_value=0)
    
    # Scale the data
    df_scaled = scaler.transform(df)
    
    # Predict stress level
    prediction = model.predict(df_scaled)
    predicted_stress_level = prediction[0][0] * 100  # Assuming output is between 0 and 1
    
    return jsonify({'predicted_stress_level': predicted_stress_level})

if __name__ == '__main__':
    app.run(debug=True)

