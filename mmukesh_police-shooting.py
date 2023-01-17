# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/data-police-shootings/fatal-police-shootings-data.csv",parse_dates=['date'])

df['month']=df['date'].dt.month

df['year']=df['date'].dt.year

df.head()
df.info()
# Treating missing value

df['armed'].fillna(df['armed'].mode()[0],inplace=True)

df['age'].fillna(df['age'].mode()[0],inplace=True)

df['gender'].fillna(df['gender'].mode()[0],inplace=True)

df['race'].fillna(df['race'].mode()[0],inplace=True)

df['flee'].fillna(df['flee'].mode()[0],inplace=True)
df.isna().sum()
#rate of shootings

df['manner_of_death'].value_counts()

gun_shot=df[df['manner_of_death']=='shot']



rate_of_gunshot=len(gun_shot['manner_of_death'])/(len(df['manner_of_death']))*100

rate_of_gunshot
#Yearswise Shooting



yearwise_shooting=df.groupby(['year'])['manner_of_death'].count()

print(yearwise_shooting)



yearwise_shooting.plot(kind='bar')

plt.xlabel("Years")

plt.ylabel("Number of Death")
# Top 10 states have the most kills

statewise_most_kills=df.groupby(['state'])['manner_of_death'].count().sort_values(ascending=False)[0:10]

statewise_most_kills.plot(kind='bar')

plt.xlabel("States")

plt.ylabel("Number of Death")
# Top 10 City have the most kills

citywise_most_kills=df.groupby(['city'])['manner_of_death'].count().sort_values(ascending=False)[0:10]

citywise_most_kills.plot(kind='bar')

plt.xlabel("City")

plt.ylabel("Number of Death")
#rate of killings relative to race and age

kill_race_age=df.groupby(['race','manner_of_death'])['age'].mean()

kill_race_age