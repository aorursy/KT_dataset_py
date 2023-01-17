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
train=pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TrainData.csv")
test=pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv")
test.head()
train.head()
train=train.drop(['Loan ID','Customer ID'],1)
test=test.drop(['Unnamed: 2','Loan ID','Customer ID'],1)
train.isna().sum()
train['Credit Score'].fillna((train['Credit Score'].mean()),inplace=True)

train['Annual Income'].fillna((train['Annual Income'].mean()),inplace=True)

train['Months since last delinquent'].fillna((train['Months since last delinquent'].mean()),inplace=True)

train['Bankruptcies'].fillna((train['Bankruptcies'].mean()),inplace=True)

train['Tax Liens'].fillna((train['Tax Liens'].mean()),inplace=True)
train.isna().sum()
def signremove(h):

    h=(str(h).strip("years"))

    h=(str(h).replace("+",""))

    h=(str(h).strip("<"))

    return float(h)
train['Years in current job']=train['Years in current job'].apply(signremove)
train['Years in current job'].fillna((train['Years in current job'].mean()),inplace=True)
train.isna().sum()
test['Credit Score'].fillna((test['Credit Score'].mean()),inplace=True)

test['Annual Income'].fillna((test['Annual Income'].mean()),inplace=True)

test['Months since last delinquent'].fillna((test['Months since last delinquent'].mean()),inplace=True)

test['Bankruptcies'].fillna((test['Bankruptcies'].mean()),inplace=True)

test['Tax Liens'].fillna((test['Tax Liens'].mean()),inplace=True)
test['Years in current job']=test['Years in current job'].apply(signremove)
test['Years in current job'].fillna((test['Years in current job'].mean()),inplace=True)
train= pd.get_dummies(train, columns=['Loan Status'],drop_first=True)

test = pd.get_dummies(test, columns=['Term','Home Ownership','Purpose'],drop_first=True)

train = pd.get_dummies(train, columns=['Term','Home Ownership','Purpose'],drop_first=True)
def removedollar(a):

    a=(str(a).replace("$"," "))

    return a
train['Monthly Debt'] = train['Monthly Debt'].apply(removedollar)

test['Monthly Debt'] = test['Monthly Debt'].apply(removedollar)
g = pd.Series(train['Maximum Open Credit'])

train['Maximum Open Credit']=pd.to_numeric(g, errors='coerce')

f = pd.Series(test['Maximum Open Credit'])

test['Maximum Open Credit']=pd.to_numeric(f, errors='coerce')
train.isna().sum()
train['Maximum Open Credit'].fillna((train['Maximum Open Credit'].mean()),inplace=True)

test['Maximum Open Credit'].fillna((test['Maximum Open Credit'].mean()),inplace=True)
from sklearn.naive_bayes import GaussianNB
gauss = GaussianNB()
remain=train.drop('Loan Status_Fully Paid',axis= 1)

loanstatus=train['Loan Status_Fully Paid']

gauss.fit(remain,loanstatus)

prediction = gauss.predict(test)
prediction

test = pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv")
predictiondata=pd.DataFrame(test['Loan ID'])
predictiondata['Loan Status']=prediction
predictiondata
predictiondata.to_csv("Harshwardhan_Palav.csv")