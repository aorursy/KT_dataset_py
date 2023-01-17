import numpy as np 

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TrainData.csv")

train = train.drop('Loan ID',1)

train = train.drop('Customer ID',1)

train= train.drop('Months since last delinquent',1)

train = train.drop('Years in current job',1)
test = pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv")

test= test.drop('Customer ID',1)

test= test.drop('Loan ID',1)

test= test.drop('Months since last delinquent',1)

test= test.drop('Unnamed: 2',1)

test = test.drop('Years in current job',1)
def data_cleaning(data):

    pd_data_Cleaning=data

    pd_data_Cleaning['Credit Score'].fillna(pd_data_Cleaning['Credit Score'].mean(),inplace=True)

    pd_data_Cleaning['Annual Income'].fillna(pd_data_Cleaning['Annual Income'].median(),inplace=True)

    pd_data_Cleaning["Maximum Open Credit"].replace('[a-zA-Z@_!#$%^&*()<>?/\|}{~:]',"0",regex=True,inplace=True)

    convert_dict = {'Maximum Open Credit': float} 

    pd_data_Cleaning[pd_data_Cleaning.Bankruptcies.isna()==True]

    pd_data_Cleaning.Bankruptcies.fillna(0.0,inplace=True)

    pd_data_Cleaning['Tax Liens'].fillna(0.0,inplace=True)

    pd_data_Cleaning["Monthly Debt"].replace('[^0-9.]',"",regex=True,inplace=True )

 

    return pd_data_Cleaning
train = data_cleaning(train)

train.isnull().sum()
test = data_cleaning(test)

test.isnull().sum()
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC as SVM

from sklearn.metrics import classification_report
y = train["Loan Status"]

x_categorical = train.loc[:,('Term','Home Ownership','Purpose','Number of Open Accounts')]

x_rest = train.drop(columns = ['Loan Status','Term','Home Ownership','Purpose','Number of Open Accounts'])

x_categorical_one_hot = pd.get_dummies(x_categorical)
x_categorical_test = test.loc[:,('Term','Home Ownership','Purpose','Number of Open Accounts')]

x_rest_test = test.drop(columns = ['Term','Home Ownership','Purpose','Number of Open Accounts'])

x_categorical_one_hot_test = pd.get_dummies(x_categorical_test)
labels = list(x_rest)

mm = MinMaxScaler()

x_scaled = pd.DataFrame(mm.fit_transform(x_rest), columns=labels)
labels = list(x_rest_test)

mm = MinMaxScaler()

x_scaled_test = pd.DataFrame(mm.fit_transform(x_rest_test), columns=labels)
X = pd.concat([x_scaled, x_categorical_one_hot], axis = 1)
X_test = pd.concat([x_rest_test, x_categorical_one_hot_test], axis = 1)
lr = LogisticRegression()

model = lr.fit(X,y)

y_pred_lr = lr.predict(X_test)
svm = SVM()

svm.fit(X, y)

y_pred_svm = svm.predict(X_test)
sub = pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv")
xx = sub['Loan ID']
csv = pd.DataFrame({'Loan ID':xx,'Loan Status':y_pred_lr})

csv.to_csv('submit5.csv')