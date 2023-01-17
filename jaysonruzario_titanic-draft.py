# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

import time

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBRegressor

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

#verticalStack = pd.concat([surveySub, surveySubLast10], axis=0)

#combined_df= pd.concat([train_df,test_df],axis=0,sort=False)
sns.heatmap(train_df.isnull())
train_df.Age.isnull().sum(),train_df.shape

test_df.Age.isnull().sum(),train_df.shape

import math

math.ceil(train_df.Age.mean(skipna=True))
train_df.Age.fillna(value= 30  ,inplace=True)

test_df.Age.fillna(value= 30  ,inplace=True)

test_df.Fare.fillna(value=36,inplace=True)
train_df.head()
# droping string values that is difficult to process

train_df.drop(['Cabin','Ticket','Name','PassengerId'],axis=1,inplace=True)

test_df.drop(['Cabin','Ticket','Name','PassengerId'],axis=1,inplace=True)

train_df.dtypes
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

train_df['Sex']=le.fit_transform(train_df.Sex)

test_df['Sex']=le.fit_transform(test_df.Sex)

#combined_df['Embarked']=le.fit_transform(combined_df.Embarked)
train_df.head()
train_df.Embarked.value_counts()
train_df=train_df.merge(pd.get_dummies(train_df.Embarked,prefix='Embarked'),how='right',on=pd.get_dummies(train_df.Embarked,prefix='Embarked').index)

test_df=test_df.merge(pd.get_dummies(test_df.Embarked,prefix='Embarked'),how='right',on=pd.get_dummies(test_df.Embarked,prefix='Embarked').index)
train_df.shape
test_df.shape
train_df.head()
train_df.drop(['key_0','Embarked'],axis=1,inplace=True)

test_df.drop(['key_0','Embarked'],axis=1,inplace=True)
train_df.head()
X=train_df.drop('Survived',axis=1)

y=train_df.Survived
from sklearn.model_selection import  train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models =[DecisionTreeClassifier,GaussianNB,RandomForestClassifier,SVC,LogisticRegression,KNeighborsClassifier,SGDClassifier]
for model in models :

    pipeline = make_pipeline( model())



    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)



    print('-------------',model,'-----------')

    print(classification_report(predictions,y_test))

    print('MAE:',mean_absolute_error(predictions,y_test))

    
pipeline_rf = make_pipeline( RandomForestClassifier(random_state=42)

                           )



pipeline_rf.fit(X_train, y_train)

predictions_rf = pipeline_rf.predict(X_test)

    

print(classification_report(predictions_rf,y_test))

print('MAE:',mean_absolute_error(predictions_rf,y_test))
train_x=train_df.drop('Survived',axis=1)



train_y=train_df.Survived
pipeline_final = make_pipeline( RandomForestClassifier(random_state=42)

                           )



pipeline_final.fit(train_x, train_y)

predictions_final = pipeline_final.predict(test_df)

    

#print(classification_report(predictions_final,y_test))

#print('MAE:',mean_absolute_error(predictions_final,y_test))
test_df = pd.read_csv('../input/test.csv')

output = pd.DataFrame({'PassengerId': test_df.PassengerId,'Survived': predictions_final})

output.to_csv('submission.csv', index=False)