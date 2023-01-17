#import all libraries for the analysis

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix 
import os

print(os.listdir("../input"))
#read input dataset 

tit_ds = pd.read_csv('../input/train.csv')

#view first few data from the dataframe



tit_ds.head(10)
print(tit_ds['Survived'].value_counts(normalize=True)*100)
sns.countplot(x='Survived', hue='Sex',data=tit_ds) 
sns.countplot(x='Survived', hue='Pclass',data=tit_ds) 
sns.countplot(x='Survived', hue='Embarked',data=tit_ds) 
tit_ds.isnull().sum()
replace_columns_NaN=['Age'];



for column in replace_columns_NaN:

    mean=tit_ds[column].mean(skipna=True)

    tit_ds[column]=tit_ds[column].replace(np.NaN,mean)
tit_ds.drop('Cabin', axis=1,inplace=True)
tit_ds.isnull().sum()
sex_mapping = {"male": 1, "female": 2}

tit_ds['Sex']=tit_ds['Sex'].map(sex_mapping)
embark_mapping = {"S": 1, "C": 2, "Q":3}

tit_ds['Embarked']=tit_ds['Embarked'].map(embark_mapping)







tit_ds['Embarked']=tit_ds['Embarked'].replace(np.NaN,1)
tit_ds.head(10)



tit_ds.isnull().sum()
tit_ds.drop(["PassengerId","Name","Ticket"],axis=1,inplace=True)
tit_ds.head(10)
x=tit_ds.drop(['Survived'], axis=1)

y=tit_ds['Survived']

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.3, random_state=0)
#Normalize train and test inputsx=tit_ds.drop(['Survived'], axis=1)

y=tit_ds['Survived']

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.3, random_state=0)
normalize_tit_ds=StandardScaler()

X_train=normalize_tit_ds.fit_transform(X_train)

X_test=normalize_tit_ds.fit_transform(X_test)

DataModel=LogisticRegression()

DataModel.fit(X_train,y_train)
PredictModel=DataModel.predict(X_test)
AccuracyModel = print(accuracy_score(PredictModel, y_test)*100)
confusion_matrix(y_test,PredictModel)