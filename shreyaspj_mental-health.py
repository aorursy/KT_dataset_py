import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings as ws

ws.filterwarnings("ignore")
data=pd.read_csv('../input/mental-health-in-tech-survey/survey.csv')
data.shape
data.head(5)
data.isnull().sum()
data.drop(columns=['state','comments'],inplace=True)
print(data['self_employed'].value_counts())

data['self_employed'].fillna('No',inplace=True)
print(data['work_interfere'].value_counts())

data['work_interfere'].fillna('Sometimes',inplace=True)
data.isnull().sum()
data.columns
sns.boxplot(data['Age'])
data.drop(data[data['Age'] < 0].index, inplace = True) 

data.drop(data[data['Age'] > 100].index, inplace = True) 
data.shape
sns.boxplot(data['Age'])
data['Gender']=[m.lower() for m in data['Gender']]

data['Gender'].value_counts()
plt.pie(data['Country'].value_counts(),labels=data['Country'].unique(),labeldistance=1.1)
country=data.groupby(data['Country'])
country['Age'].aggregate(np.median).sort_values()
country['treatment','remote_work','self_employed'].describe()
s_employ=data.groupby(['self_employed'])

s_employ['treatment'].describe()
treat=data.groupby(['treatment'])

treat['Age'].describe()
data.columns
treat['tech_company','work_interfere'].describe()
sns.barplot(data['leave'].unique(),data['leave'].value_counts())
sns.barplot(data['mental_health_consequence'].unique(),data['mental_health_consequence'].value_counts())
sns.barplot(data['phys_health_consequence'].unique(),data['phys_health_consequence'].value_counts())
plt.pie(data['coworkers'].value_counts(),labels=data['coworkers'].unique())

data['coworkers'].value_counts()
print(data['mental_vs_physical'].value_counts())

plt.hist(data['mental_vs_physical'],histtype='step')