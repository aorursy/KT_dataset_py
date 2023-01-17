import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
shots=pd.read_csv('../input/data-police-shootings/fatal-police-shootings-data.csv')

shots.info()
shots.describe(include='all')
shots.head()
shots.isna().sum()
shots.dropna(subset=['armed'], inplace=True)

shots.dropna(subset=['age'], inplace=True)

shots.dropna(subset=['gender'], inplace=True)

shots.dropna(subset=['race'], inplace=True)

shots.dropna(subset=['flee'], inplace=True)

shots.isna().sum()
shots['date']=pd.to_datetime(shots['date'])

shots['Year']=shots['date'].dt.year

shots.head()
yshot=shots['Year'].value_counts()

yearlyshots=yshot.to_frame()

yearlyshots.reset_index(level=0, inplace=True)

yearlyshots.columns=['Year','No of shots']

yearlyshots.head()
plt.bar(yearlyshots['Year'],yearlyshots['No of shots'])

plt.xlabel='Year'

plt.ylabel='No of Shots'
Gendershots=shots['gender'].value_counts()

gendershots=Gendershots.to_frame()

gendershots.reset_index(level=0, inplace=True)

gendershots.columns=['gender','shots']

gendershots.head()
sns.barplot(x='gender',y='shots',data=gendershots)
rshots=shots['race'].value_counts()

raceshots=rshots.to_frame()

raceshots.reset_index(level=0, inplace=True)

raceshots.columns=['Race','shots']

raceshots.head()
sns.barplot(x='Race',y='shots',data=raceshots)
shots['armed or not'] = shots['armed'].apply(lambda x: 'F'  if x == 'unarmed' else 'T')

ashots=shots['armed or not'].value_counts()

armshots=ashots.to_frame()

armshots.reset_index(level=0, inplace=True)

armshots.columns=['Armed or not','shots']

armshots.head()
sns.barplot(x='Armed or not',y='shots',data=armshots)
sns.catplot(x="gender", y="age", hue="threat_level", kind="bar", data=shots);
sns.catplot(x="gender", y="age", hue="signs_of_mental_illness", kind="bar", data=shots);
sns.boxplot(x='gender',y='age',data=shots);