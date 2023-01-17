import numpy as np

import pandas as pd

import csv

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

df=pd.read_csv('/kaggle/input/heart-disease-prediction-using-logistic-regression/framingham.csv')

df.head(30)
df_fill= df

sns.countplot(x='male',hue='male',data=df)
sns.countplot(x='education',hue='education',data=df)
sns.countplot(x='currentSmoker',hue='currentSmoker',data=df)
sns.countplot(x='cigsPerDay',data=df)
sns.countplot(x='currentSmoker',hue='currentSmoker',data=df)
sns.countplot(x='totChol',data=df)
sns.countplot(x='prevalentStroke',hue='prevalentStroke',data=df)
sns.countplot(x='prevalentHyp',data=df)
sns.countplot(x='BMI',data=df)
sns.countplot(x='diabetes',hue='diabetes',data=df)
sns.countplot(x='diaBP',data=df)
sns.countplot(x='sysBP',data=df)
sns.countplot(x='glucose',data=df)
sns.countplot(x='BPMeds',data=df)
sns.countplot(x='heartRate',data=df)
sns.heatmap(df)
sns.countplot(x='TenYearCHD',hue='TenYearCHD',data=df)
df.info()
df.isnull()
df.sum()
df.head(5)
df['totChol'].unique()
df.drop(['male','age','education', 'currentSmoker','cigsPerDay'],axis=1,inplace=True)
df.head(438)
x=df["diabetes"].values.reshape(-1,1)

y=df["TenYearCHD"].values.reshape(-1,1)
df.isnull().sum()
missing_val_count_by_column = (df.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
df.info()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=df,copy=True)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=0)
from sklearn.linear_model import LogisticRegression 
logmodel=LogisticRegression()
import warnings

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
logmodel.fit(x_train,y_train)
pred=logmodel.predict(x_test)
from sklearn.metrics import classification_report
classification_report(x_test,pred)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)