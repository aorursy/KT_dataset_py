import numpy as npa

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/heart.csv')
#Check first few rows in the data

df.head()
df.shape
df.info()
df.replace({"sex": {1: 'Male', 0:'Female'}}, inplace=True)   

df.replace({"target":{1:'Yes', 0:'No'}}, inplace=True)
df['sex'].value_counts()
sns.countplot(x=df['sex'],data=df)
df[(df['sex']=='Male') & (df['target']=='Yes')].shape
df[(df['sex']=='Female') & (df['target']=='Yes')].shape
sns.countplot(x=df['sex'],hue=df['target'],data=df)
sns.set(color_codes=True)

sns.distplot(df['age'],bins=20,kde=False)
male_data=df[(df['sex']=='Male') & (df['target']=='Yes')]

sns.distplot(male_data['age'],bins=20,kde=False)
female_data=df[(df['sex']=='Female') & (df['target']=='Yes')]
sns.distplot(female_data['age'],bins=20,kde=False)
positive_cases=df[df['target']=='Yes']

sns.jointplot(positive_cases['chol'],positive_cases['trestbps'],kind='reg')
sns.countplot(x=positive_cases['target'],hue=positive_cases['fbs'],data=positive_cases)
sns.countplot(x=positive_cases['exang'], hue=positive_cases['target'],data=positive_cases)
sns.distplot(positive_cases['chol'],bins=20,kde=True)
sns.countplot(x=positive_cases['slope'], data=positive_cases)
corr_var=positive_cases[['age', 'trestbps', 'chol','thalach','oldpeak']]

cv=corr_var.corr()

cv
sns.heatmap(cv,cmap='coolwarm',annot=True)