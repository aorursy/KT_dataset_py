# Import all neccessary libraties

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
# Load data

df = pd.read_csv("/kaggle/input/dataloan/dataloan.csv")

df
# Total shape

df.shape
# Rows

df.shape[0]
# Column 

df.shape[1]
# First five records

df.head(5)
# Last five records

df.tail(5)
# Data visualization

sns.set_style("whitegrid")

sns.countplot(x="Loan_Status", hue="Gender", data=df,)
sns.countplot(x="Loan_Status", hue="Education", data=df,)
sns.countplot(x="Loan_Status", hue="Self_Employed", data=df,)
sns.countplot(x="Loan_Status", hue="Property_Area", data=df,)
# Check NAN values

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# Fill NAN values

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)

df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

df['Credit_History'].fillna(df['Credit_History'].mean(), inplace=True)
# Descibe data

df.describe()
# data info

df.info()
# Dependent variable

y = df['Loan_Status']
# Independent variable

x = df.drop(['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','CoapplicantIncome','Property_Area','Loan_Status'], axis=1)
# Split data into 70:30



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.3, random_state=0)
#Logistic regression

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(solver='lbfgs')
# Fit data

logmodel.fit(x_train, y_train)
# Predict result

prediction = logmodel.predict(x_test)
# Classification Report (CR)

from sklearn.metrics import classification_report

classification_report(y_test,prediction)
# Confusion Matrix (CM)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, prediction)
# Accuracy Score (AS)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, prediction)