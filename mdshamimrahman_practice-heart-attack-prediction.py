
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline


df=pd.read_csv('../input/heart-disease-prediction-using-logistic-regression/framingham.csv')
df.head(5)
#Check Null Value

df.isnull().sum()
##Handle Null Value
mean=df['education'].mean()
mean
#Replace Null into 1
df['education']=df.education.fillna(mean)
mean_cig=df['cigsPerDay'].mean()
mean_cig
df['cigsPerDay']=df['cigsPerDay'].fillna(mean_cig)
median_bp=df['BPMeds'].median()
df['BPMeds']=df['BPMeds'].fillna(median_bp)
col_mean=df['totChol'].mean()
df['totChol']=df['totChol'].fillna(col_mean)
heart_mean=df['heartRate'].mean()
df['heartRate']=df['heartRate'].fillna(heart_mean)
glucose_mean=df['glucose'].mean()
df['glucose']=df['glucose'].fillna(glucose_mean)
df.isnull().sum()
##How Many columns and Row
df.shape
df.head(5)
#how many yes or no in 	TenYearCHD

df['TenYearCHD'].value_counts()
## Splited features and label

x=df.drop('TenYearCHD',axis=1)

x.head(2)
y=df['TenYearCHD']
##Splited test train Data 
from sklearn.model_selection import train_test_split


xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=.20)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
x.info()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x=x.drop('BMI',axis=1)
x.info()
x.isnull().sum()
lr.fit(x,y)
y.isnull().sum()
lr.score(x,y)
lr.predict(x)
