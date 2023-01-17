# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")
test = pd.read_csv("/kaggle/input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv")
train
test
train.isnull().sum()
test.isnull().sum()
train.dropna(inplace=True)
test.dropna(inplace=True)
train['Loan_Status'].unique()
gender_mapping = {'Male' : 0, 'Female' : 1}
married_mapping = {'No' : 0, 'Yes' : 1}
education_mapping = {'Graduate' : 0, 'Not Graduate' : 1}
area_mapping = {'Rural' : 0, 'Urban' : 1, 'Semiurban' : 2}
status_mapping = {'N' : 0, 'Y' : 1}
train['Gender'] = train['Gender'].map(gender_mapping)
test['Gender'] = test['Gender'].map(gender_mapping)

train['Married'] = train['Married'].map(married_mapping)
test['Married'] = test['Married'].map(married_mapping)

train['Education'] = train['Education'].map(education_mapping)
test['Education'] = test['Education'].map(education_mapping)

train['Property_Area'] = train['Property_Area'].map(area_mapping)
test['Property_Area'] = test['Property_Area'].map(area_mapping)

train['Self_Employed'] = train['Self_Employed'].map(married_mapping)
test['Self_Employed'] = test['Self_Employed'].map(married_mapping)

train['Dependents'] = train['Dependents'].map(dependence_mapping)
test['Dependents'] = test['Dependents'].map(dependence_mapping)

train['Loan_Status'] = train['Loan_Status'].map(status_mapping)
train
test
x_train = train.drop(['Loan_Status', 'Loan_ID'], axis=1)
y_train = train['Loan_Status']

x_test = test.drop('Loan_ID', axis=1)
x_train
x_test
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
pred
PREDICTIONS = pd.DataFrame({'Loan_ID' : test['Loan_ID'], 'Predictions' : pred})
PREDICTIONS
