import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px

%matplotlib inline

data = pd.read_csv('../input/hackerearth-employee-attrition-rate/Train.csv')
submission_data = pd.read_csv('../input/hackerearth-employee-attrition-rate/Test.csv')
data.shape
submission_data.shape
data.head()
data.describe()
data.columns
features = ['Age', 'Compensation_and_Benefits', 'Work_Life_balance', 'Post_Level', 'growth_rate', 'Time_of_service', 'Pay_Scale', 'Hometown', 'Education_Level']
data[features].isna().sum()
# Convert the categorical data into numaric value

for feature in features:
    if data[feature].dtype == 'object':
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].cat.codes
data['Age'].fillna(52, inplace=True)
data['Work_Life_balance'].fillna(3, inplace=True)
data['Time_of_service'].fillna(21, inplace=True) 
data['Pay_Scale'].fillna(8, inplace=True)
data[features].isna().sum()
submission_data['Age'].fillna(52, inplace=True)
submission_data['Work_Life_balance'].fillna(3, inplace=True)
submission_data['Time_of_service'].fillna(21, inplace=True) 
submission_data['Pay_Scale'].fillna(8, inplace=True)
submission_data[features].isna().sum()
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
X, y = data[features].values, data['Attrition_rate'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

def prepare_inputs(X_train, X_test):
    ohe = OrdinalEncoder()
    ohe.fit(X_train)
    X_train_enc = ohe.transform(X_train)
    X_test_enc = ohe.transform(X_test)
    return X_train_enc, X_test_enc

X_train, X_test = prepare_inputs(X_train, X_test)
from sklearn.linear_model import LinearRegression
from sklearn import metrics


model = LinearRegression()
model.fit(X_train, y_train)

print('Intercept: \n', model.intercept_)
print('Coefficients: \n', model.coef_)
output = model.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': output.flatten()})
df
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, output))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, output))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, output)))
XX = submission_data[features].values

ohe = OrdinalEncoder()
ohe.fit(XX)
XX = ohe.transform(XX)
y_predict = model.predict(XX)

import csv

with open('5th_submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Employee_ID", "Attrition_rate"])
    
    for i in range(3000):
        writer.writerow([submission_data['Employee_ID'][i], y_predict[i]])