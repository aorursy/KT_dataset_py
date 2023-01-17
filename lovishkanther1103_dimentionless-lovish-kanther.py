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
import numpy as np

import pandas as pd
train=pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TrainData.csv")

test=pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv")
train.head()
train=train.drop(['Loan ID','Customer ID'],1)
train.isna().sum()
test.head()
test=test.drop(['Loan ID','Customer ID','Unnamed: 2'],1)
def rem(x):

    x=(str(x).replace("$",""))

    return float(x)



train['Monthly Debt']=train['Monthly Debt'].apply(rem)

test['Monthly Debt']=test['Monthly Debt'].apply(rem)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

lb = LabelBinarizer()

train['Loan Status'] = lb.fit_transform(train['Loan Status'])
train['Credit Score']=train['Credit Score'].fillna((train['Credit Score'].mean()))

train['Annual Income']=train['Annual Income'].fillna((train['Annual Income'].mean()))

train['Months since last delinquent']=train['Months since last delinquent'].fillna((train['Months since last delinquent'].mean()))
train.isna().sum()
train.head()
train['Bankruptcies'].unique()
train['Tax Liens'].unique()
train['Bankruptcies']=train['Bankruptcies'].fillna(int((train['Bankruptcies'].mean())))

train['Tax Liens']=train['Tax Liens'].fillna(int(train['Tax Liens'].mean()))
train.isna().sum()
train['Years in current job'].unique()
import re

def getnum(a):

    a=(str(a).strip("years"))

    a=(str(a).replace("+",""))

    a=(str(a).strip("<"))

    return float(a)

    
train['Years in current job']=train['Years in current job'].apply(getnum)
train['Years in current job']=train['Years in current job'].fillna(int((train['Years in current job'].mean())))
train.rename(columns={"Maximum Open Credit":"Maximum_Open_Credit"},inplace=True)

test.rename(columns={"Maximum Open Credit":"Maximum_Open_Credit"},inplace=True)
q = pd.Series(train['Maximum_Open_Credit'])

train['Maximum_Open_Credit']=pd.to_numeric(q,errors="coerce")
a = pd.Series(test['Maximum_Open_Credit'])

test['Maximum_Open_Credit']=pd.to_numeric(a, errors="coerce")
train['Maximum_Open_Credit']=train['Maximum_Open_Credit'].fillna((train['Maximum_Open_Credit'].mean()))

test['Maximum_Open_Credit']=test['Maximum_Open_Credit'].fillna((test['Maximum_Open_Credit'].mean()))
train.dtypes
train.isna().sum()
test.isna().sum()
test['Credit Score']=test['Credit Score'].fillna((test['Credit Score'].mean()))

test['Annual Income']=test['Annual Income'].fillna((test['Annual Income'].mean()))

test['Months since last delinquent']=test['Months since last delinquent'].fillna((test['Months since last delinquent'].mean()))

test['Bankruptcies']=test['Bankruptcies'].fillna(int((test['Bankruptcies'].mean())))

test['Tax Liens']=test['Tax Liens'].fillna(int(test['Tax Liens'].mean()))

test['Years in current job']=test['Years in current job'].apply(getnum)

test['Years in current job']=test['Years in current job'].fillna(int((test['Years in current job'].mean())))
test.isna().sum()
train=pd.get_dummies(train,columns=['Home Ownership','Term','Purpose'])

test=pd.get_dummies(test,columns=['Home Ownership','Term','Purpose'])
train.shape
test.shape
train_y=train['Loan Status']

train_x=train.drop('Loan Status',axis= 1)
RF = RandomForestClassifier()
RF.fit(train_x,train_y)
pred = RF.predict(test)
pred
test=pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv")
y_predict=pd.DataFrame(test['Loan ID'])
y_predict['Loan Status']=pred
y_predict
y_predict.to_csv("Dimentionless_Lovish_Kanther.csv")