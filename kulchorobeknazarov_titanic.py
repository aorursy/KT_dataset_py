import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

sns.set()
df=pd.read_csv("../input/titanic/train.csv")

df.isnull().sum()

no_m_v=df.dropna(axis=0)

cleaned_data=no_m_v.drop(['Name','Ticket','Cabin'],axis=1)

data_w_d=pd.get_dummies(cleaned_data,drop_first=True)
targets=data_w_d['Survived']

inputs=data_w_d.drop(['Survived'],axis=1)
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier()

clf.fit(inputs,targets)

clf.score(inputs,targets)
from sklearn.model_selection import cross_val_score

validation=cross_val_score(clf,inputs,targets,cv=5)

validation
test_df=pd.read_csv('../input/titanic/test.csv')

test_df.isnull().sum()

no_m_v2=test_df.dropna(axis=0)

cleaned=no_m_v2.drop(['Name','Ticket','Cabin'],axis=1)

test_data=pd.get_dummies(cleaned,drop_first=True)

clf.predict(test_data)
test_data['Survived']=clf.predict(test_data)
test_data.to_csv('Titanic',index=False)