import numpy as np 

import pandas as pd 

import sklearn 

import seaborn as sns

import matplotlib.pyplot as plt
test=pd.read_csv('../input/test.csv')

train=pd.read_csv('../input/train.csv')
train.head()
for df in [train,test]:

    df.set_index('Id',inplace=True)
test.head()
train.head()
X_test=test

X_train=train.drop(columns='target')

y_train=train['target']
sns.countplot(train['sex'])
sns.countplot(train['race'])
sns.countplot(train['workclass'])
from scipy import stats #mecanismo para descobrir a moda



for A in X_train.columns:

    for data in [X_train,X_test]:

        data[A]=data[A].replace(' ?',stats.mode(data[A])[0][0])

        
One_Hot_Xtrain = pd.get_dummies(X_train)

One_Hot_Xtest = pd.get_dummies(X_test)

X_train, X_test = One_Hot_Xtrain.align(One_Hot_Xtest,join='left',axis=1)
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.transform(X_test)
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

X_train = my_imputer.fit_transform(X_train)

X_test = my_imputer.transform(X_test)
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier 
clf=RandomForestClassifier(random_state=100,min_samples_leaf=5,min_samples_split=3,max_features=None)
clf.fit(X_train,y_train)
y_predict=clf.predict(X_test)
y_predict
Submission=pd.DataFrame()
Submission['Id']=test.index

Submission['target']=y_predict
Submission.head()
Submission.to_csv('submission2.csv',index=False)