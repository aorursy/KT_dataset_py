# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import sklearn



df_train = pd.read_csv('../input/fraud-csv/fraud.csv')
df_train.shape
df_train.columns
# df_train = df_train.dropna(axis=0)
df_train.head()
df_test = pd.read_csv('../input/fraud-detection/fraud_test.csv')
df_test.shape
df_test.columns = ['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'ununnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed','unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed',  'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed' ,'unnamed','unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed','unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed','unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed', 'unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed','unnamed', 'unnamed', 'unnamed','unnamed', 'unnamed',]
# df_test = df_test.dropna(axis=0)
df_test.columns
df_test.head()
# Modeling

y = df_train.isFraud
df_train_features = ['TransactionID', 'TransactionDT', 'TransactionAmt', 'addr1', 'P_emaildomain']
X = df_train[df_train_features]
X.describe()
X.head(5)
from sklearn.tree import DecisionTreeRegressor



# Define model. Specify a number for random_state to ensure same results each run

train_model = DecisionTreeRegressor(random_state=1)



# Fit model

train_model.fit(X, y)