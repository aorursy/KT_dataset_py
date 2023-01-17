# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

train = pd.read_csv('/kaggle/input/credit-risk-modeling-case-study/CRM_TrainData.csv')

train.head()
test = pd.read_csv('/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv')

test.head()
train['Loan Status'].unique()
train = train.drop(['Loan ID','Customer ID'],1)

train.head()
train.shape
train.isna().sum()
train['Credit Score'].fillna((train['Credit Score'].mean()),inplace=True)

train['Annual Income'].fillna((train['Annual Income'].mean()),inplace=True)

train['Months since last delinquent'].fillna((train['Months since last delinquent'].mean()),inplace=True)

train['Bankruptcies'].fillna((train['Bankruptcies'].mean()),inplace=True)

train['Tax Liens'].fillna((train['Tax Liens'].mean()),inplace=True)
train.isna().sum()
def strip(a):

    a=(str(a).strip("years"))

    a=(str(a).replace("+",""))

    a=(str(a).strip("<"))

    return float(a)
train['Years in current job']=train['Years in current job'].apply(strip)
train['Years in current job'].fillna((train['Years in current job'].mean()),inplace=True)
train.isna().sum()
test = test.drop(['Loan ID','Customer ID','Unnamed: 2'],1)

test.head()
test.isna().sum()
test['Credit Score'].fillna((test['Credit Score'].mean()),inplace=True)

test['Annual Income'].fillna((test['Annual Income'].mean()),inplace=True)

test['Months since last delinquent'].fillna((test['Months since last delinquent'].mean()),inplace=True)

test['Bankruptcies'].fillna((test['Bankruptcies'].mean()),inplace=True)

test['Tax Liens'].fillna((test['Tax Liens'].mean()),inplace=True)
test['Years in current job']=test['Years in current job'].apply(strip)
test['Years in current job'].fillna((test['Years in current job'].mean()),inplace=True)
test.isna().sum()
train.dtypes
train = pd.get_dummies(train, columns=['Term','Home Ownership','Purpose'],drop_first=True)

train.head()
train.shape
test = pd.get_dummies(test, columns=['Term','Home Ownership','Purpose'],drop_first=True)

test.head()
test.shape
train= pd.get_dummies(train, columns=['Loan Status'],drop_first=True)

train.head()
train.shape
def dollar(a):

    a=(str(a).replace("$"," "))

    return a
train['Monthly Debt'] = train['Monthly Debt'].apply(dollar)
test['Monthly Debt'] = test['Monthly Debt'].apply(dollar)
x = pd.Series(train['Maximum Open Credit'])

train['Maximum Open Credit']=pd.to_numeric(x, errors='coerce')
y = pd.Series(test['Maximum Open Credit'])

test['Maximum Open Credit']=pd.to_numeric(y, errors='coerce')
train['Maximum Open Credit'].fillna((train['Maximum Open Credit'].mean()),inplace=True)

test['Maximum Open Credit'].fillna((test['Maximum Open Credit'].mean()),inplace=True)
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier
lr = LogisticRegression()

gnb = GaussianNB()

rfr = RandomForestClassifier()
x_train=train.drop('Loan Status_Fully Paid',axis= 1)

y_train=train['Loan Status_Fully Paid']

rfr.fit(x_train,y_train)
y_pred = rfr.predict(test)
y_pred
test = pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv")
y_pred_rf=pd.DataFrame(test['Loan ID'])
y_pred_rf
y_pred_rf['Loan Status']=y_pred
y_pred_rf
y_pred_rf.to_csv("Devesh-Patodia-Dimensionless-Technologies-1.csv")