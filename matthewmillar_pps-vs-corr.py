!pip install ppscore

# import ppscore as pps
import warnings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import ppscore as pps

df_train = pd.read_csv('../input/titanic/train.csv')

df_test  = pd.read_csv('../input/titanic/test.csv')

df_sample= pd.read_csv('../input/titanic/gender_submission.csv')
df_month = pd.read_csv('../input/housing-in-london/housing_in_london_monthly_variables.csv')

df_year = pd.read_csv('../input/housing-in-london/housing_in_london_yearly_variables.csv')

print(df_month.head(1))

print(df_year.head(1))
# Remove nan

df_month = df_month.dropna()

df_year = df_year.dropna()
print(df_month.columns)

print(df_year.columns)
print(df_year.head())
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in df_train["Name"]]

df_train["Title"] = pd.Series(dataset_title)

df_train["Title"] = df_train["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df_train["Title"] = df_train["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

df_train["Title"] = df_train["Title"].astype(int)

df_train.drop(labels = ["Name"], axis = 1, inplace = True)



sex = pd.get_dummies(df_train['Sex'],drop_first=True)

embark = pd.get_dummies(df_train['Embarked'],drop_first=True)

df_train = pd.concat([df_train,sex,embark],axis=1)



df_train["Family"] = df_train["SibSp"] + df_train["Parch"] + 1

df_train['Single'] = df_train['Family'].map(lambda s: 1 if s == 1 else 0)

df_train['SmallF'] = df_train['Family'].map(lambda s: 1 if  s == 2  else 0)

df_train['MedF']   = df_train['Family'].map(lambda s: 1 if 3 <= s <= 4 else 0)

df_train['LargeF'] = df_train['Family'].map(lambda s: 1 if s >= 5 else 0)

df_train['Senior'] = df_train['Age'].map(lambda s:1 if s>60 else 0)



def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex



df_train['Person'] = df_train[['Age','Sex']].apply(get_person,axis=1)



person_dummies_train  = pd.get_dummies(df_train['Person'])

person_dummies_train.columns = ['Child','Female','Male']

person_dummies_train.drop(['Male'], axis=1, inplace=True)



df_train = df_train.join(person_dummies_train)



df_train.drop(['Person'],axis=1,inplace=True)

df_train.drop('male',axis=1,inplace=True)

df_train.drop(['Cabin','Ticket'],axis = 1, inplace= True)

df_train.drop(['Sex','Embarked'],axis=1,inplace=True)

df_train.drop(['PassengerId'],axis=1,inplace=True)



df_train.head()
warnings.filterwarnings("ignore")

# PPS needs this is a lot of warning it produces. 
plt.figure(figsize=(16,12))

sns.heatmap(pps.matrix(df_train),annot=True,fmt=".2f")
plt.figure(figsize=(16,12))

sns.heatmap(df_train.corr(),annot=True,fmt=".2f")
# Houseing for year

plt.figure(figsize=(16,12))

sns.heatmap(pps.matrix(df_year),annot=True,fmt=".2f")
plt.figure(figsize=(16,12))

sns.heatmap(df_year.corr(),annot=True,fmt=".2f")
# Houseing for year

plt.figure(figsize=(16,12))

sns.heatmap(pps.matrix(df_month),annot=True,fmt=".2f")
plt.figure(figsize=(16,12))

sns.heatmap(df_month.corr(),annot=True,fmt=".2f")