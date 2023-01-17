import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
#Importing Dataset

loan=pd.read_csv("../input/loandataset/train.csv")
loan.head()
#Dropping Loan_ID Column as its not needed

loan=loan.drop("Loan_ID", axis=1)
loan.isnull().sum()
#Checking the Value Count of Each Row to fill the missing data accordingly

gender=loan["Gender"].value_counts()

married=loan["Married"].value_counts()

dependents=loan["Dependents"].value_counts()

credit=loan["Credit_History"].value_counts()

print(gender)

print(married)

print(dependents)

print(credit)
#Filling the Missing Data

loan["LoanAmount"].fillna(loan["LoanAmount"].mean(), inplace=True)

loan["Loan_Amount_Term"].fillna(loan["Loan_Amount_Term"].median(), inplace=True)

loan["Credit_History"].fillna(2,inplace=True)

loan["Dependents"].fillna(4,inplace=True)

loan["Self_Employed"].fillna(2,inplace=True)

loan["Married"].fillna(2,inplace=True)

loan["Gender"].fillna(2,inplace=True)

loan["Dependents"].replace("3+",3,inplace=True)
lb=LabelEncoder()
#Converting Categorical Data into Numeric Data

loan["Property_Area"]=lb.fit_transform(loan["Property_Area"])

loan["Education"]=lb.fit_transform(loan["Education"])

loan["Gender"].replace("Male",1,inplace=True)

loan["Gender"].replace("Female",0,inplace=True)

loan["Married"].replace("No",0,inplace=True)

loan["Married"].replace("Yes",1,inplace=True)

loan["Self_Employed"].replace("Yes",1,inplace=True)

loan["Self_Employed"].replace("No",0,inplace=True)
x_train=loan.drop("Loan_Status", axis=1)

y_train=loan["Loan_Status"]
RF_model=RandomForestClassifier()
RF_model.fit(x_train,y_train)
RF_model.score(x_train,y_train)
#Importing Test Data

test=pd.read_csv("../input/loantestdata/loan_test.csv")

test.head()
test=test.drop("Loan_ID", axis=1)
test.isnull().sum()
test["LoanAmount"].fillna(test["LoanAmount"].mean(), inplace= True)

test["Loan_Amount_Term"].fillna(test["Loan_Amount_Term"].median(), inplace= True)

test["Dependents"].fillna(4, inplace= True)

test["Self_Employed"].fillna(2, inplace= True)

test["Credit_History"].fillna(2, inplace= True)

test["Gender"].fillna(2, inplace= True)

test["Dependents"].replace("3+",3,inplace=True)
#Converting Categorical Data into Numeric Data in Test Dataset

test["Property_Area"]=lb.fit_transform(test["Property_Area"])

test["Education"]=lb.fit_transform(test["Education"])

test["Married"]=lb.fit_transform(test["Married"])

test["Self_Employed"].replace("Yes",1,inplace=True)

test["Self_Employed"].replace("No",0,inplace=True)

test["Gender"].replace("Male",1,inplace=True)

test["Gender"].replace("Female",0,inplace=True)
Prediction=RF_model.predict(test)
print(Prediction)