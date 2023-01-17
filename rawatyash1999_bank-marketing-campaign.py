import pandas as pd

import numpy as np

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df =  pd.read_csv('/kaggle/input/bank-marketing-campaigns-dataset/bank-additional-full.csv', sep=';')

df.describe()
df.columns
df.isnull().sum()
dummy1 = pd.get_dummies(df['job'])
dfjob = dummy1.drop(['unknown'],axis=1)

dfjob.head()
dummy2 = pd.get_dummies(df['marital'])

maritaldf = dummy2.drop(['unknown'],axis=1)

maritaldf.head()
dummy3 = pd.get_dummies(df['education'])

educationdf = dummy3.drop(['unknown'],axis=1)

educationdf.head()
dummy3.isnull().sum()
dummy4 = pd.get_dummies(df['default'])

defaultdf = dummy4.drop(['yes'], axis=1)

defaultdf.head()
dummy5 = pd.get_dummies(df['housing'])

housingdf = dummy5.drop(['yes'], axis=1)

housingdf.head()
dummy6 = pd.get_dummies(df['loan'])

loandf = dummy6.drop(['yes'], axis=1)

loandf.head()
df['duration']
dummy8 = pd.get_dummies(df['poutcome'])

poutcomedf = dummy8.drop(['success'], axis=1)

poutcomedf.head()
df['y'] = df['y'].map({'yes':1,'no':0})
df
merged = pd.concat([df, dfjob, educationdf, defaultdf, loandf, poutcomedf, housingdf, maritaldf],join='outer', axis=1)
merged.head()
data = merged.drop(['marital','job', 'education', 'default', 'loan', 'housing', 'contact', 'month', 'day_of_week', 'poutcome'], axis=1)
data.head()
data.isnull().sum()
from sklearn.model_selection import train_test_split
x = data.drop(['y'],axis=1)

y = data.y
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.20,random_state=5)
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(xtrain, ytrain)
model.score(xtest, ytest)
model.intercept_
model.coef_
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

clf.fit(xtrain,ytrain, sample_weight=None, check_input=True, X_idx_sorted=None)

clf.get_params(deep=True)

clf.predict(xtest, check_input=True)

clf.predict_log_proba(xtest)

clf.predict(xtest,check_input=True)

print(clf.score(xtest,ytest, sample_weight=None))
modelNew=RandomForestClassifier(n_estimators=100)

modelNew.fit(xtrain, ytrain)
prediction = model.predict(xtest)
from sklearn import metrics
df=pd.DataFrame(prediction,ytest)

print(df)
metrics.accuracy_score(prediction,ytest)