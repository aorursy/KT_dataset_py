# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/train-loan/train_ctrUa4K.csv")

test_data = pd.read_csv("/kaggle/input/test-loan/test_lAUu6dG.csv")

sample_data = pd.read_csv("/kaggle/input/sample-loan/sample_submission_49d68Cx.csv")
y_train = train_data['Loan_Status']

x_train = train_data.drop(['Loan_Status'], axis =1)
one_hot_encoded_training_predictors = pd.get_dummies(x_train['Property_Area'])
x_train.drop(['Property_Area'],axis =1)

x_train['Property_Area'] = one_hot_encoded_training_predictors
ohe1 = pd.get_dummies(x_train['Gender'])

x_train.drop(['Gender'],axis =1)

x_train['Gender'] = ohe1

ohe2 = pd.get_dummies(x_train['Married'])

x_train.drop(['Married'],axis =1)

x_train['Married'] = ohe2

ohe3 = pd.get_dummies(x_train['Education'])

x_train.drop(['Education'],axis =1)

x_train['Education'] = ohe3
ohe4 = pd.get_dummies(x_train['Self_Employed'])

x_train.drop(['Self_Employed'],axis = 1)

x_train['Self_Employed'] = ohe4
ohe5 = pd.get_dummies(x_train['Dependents'])

x_train.drop(['Dependents'],axis = 1)

x_train['Dependents'] = ohe5
x_train1 = x_train.iloc[ : ,1:]
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

data_with_imputed_values = my_imputer.fit_transform(x_train1)
x_test = test_data
ohe6 = pd.get_dummies(x_test['Gender'])

x_test.drop(['Gender'],axis =1)

x_test['Gender'] = ohe6

ohe7 = pd.get_dummies(x_test['Married'])

x_test.drop(['Married'],axis =1)

x_test['Married'] = ohe7

ohe8 = pd.get_dummies(x_test['Education'])

x_test.drop(['Education'],axis =1)

x_test['Education'] = ohe8

ohe9 = pd.get_dummies(x_test['Self_Employed'])

x_test.drop(['Self_Employed'],axis =1)

x_test['Self_Employed'] = ohe9

ohe10 = pd.get_dummies(x_test['Property_Area'])

x_test.drop(['Property_Area'],axis =1)

x_test['Property_Area'] = ohe10

ohe11 = pd.get_dummies(x_test['Dependents'])

x_test.drop(['Dependents'],axis =1)

x_test['Dependents'] = ohe11
x_test1 = x_test.iloc[ : ,1:]
test_data_with_imputed_values = my_imputer.fit_transform(x_test1)
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import sklearn
rf = sklearn.ensemble.RandomForestClassifier()
rf.fit(data_with_imputed_values,y_train)
y_pred2 = rf.predict(test_data_with_imputed_values)
op1 = test_data.iloc[ : ,0:1]
op1['Loan_Status'] = y_pred2
op1.set_index('Loan_ID', inplace=True)
op1
import os

os.chdir('/kaggle/working')
op1.to_csv(r'op1.csv')
from IPython.display import FileLink

FileLink(r'op1.csv')