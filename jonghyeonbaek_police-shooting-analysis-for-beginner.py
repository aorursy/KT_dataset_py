# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')
data
data.isnull().sum()
age = np.array(data['age'])
nan_age = np.where(np.isnan(data['age']))
nan_age = nan_age[0]
# 결측값 표시

mean_age = int(np.mean(data['age']))

for i in range(len(nan_age)):
    data['age'][nan_age[i]] = mean_age
Gender = np.array(data['gender'])
nan_gender = np.where(data['gender'].isnull())
nan_gender = nan_gender[0]
# 결측값 표시

for i in range(len(nan_gender)):
    data['gender'][nan_gender[i]] = 'M'
data.isnull().sum()
race = set(data['race'])
race = list(race)
race
del race[0]
race
Race = np.array(data['race'])
nan_race = np.where(data['race'].isnull())
nan_race = nan_race[0]
for i in range(len(nan_race)):
    data['race'][nan_race[i]] = race[i % 6]
data.isnull().sum()
flee = set(data['flee'])
flee = list(flee)
flee
del flee[1]
flee
Flee = np.array(data['flee'])
nan_flee = np.where(data['flee'].isnull())
nan_flee = nan_flee[0]
for i in range(len(nan_flee)):
    data['flee'][nan_flee[i]] = [i % 4]
data.isnull().sum()
# 1. Shooting rate

shot_feature = set(data['manner_of_death'])
print(shot_feature)
shooting = np.where(data['manner_of_death'] == 'shot')
shot_and_Tasered = np.where(data['manner_of_death'] == 'shot and Tasered')

import seaborn as sns

sns.countplot(x="manner_of_death",data = data)
plt.show()
print("총을 쏜 비율 :" ,np.round(len(shooting[0]) / len(data['manner_of_death']),3))
print("총과 테이저를 쏜 비율 :" ,np.round(len(shot_and_Tasered[0]) / len(data['manner_of_death']),3))
# What is the rate of killings relative to race and age

plt.figure(figsize=(10,5))
sns.countplot(x="race",data = data)
plt.show()

plt.figure(figsize=(15,5))
sns.distplot(data["age"])
plt.show()
# Which states have the most kills
plt.figure(figsize=(15,5))
sns.countplot(x="state",data = data)
plt.show()
data_date = pd.to_datetime(data['date'], format='%Y-%m-%d', errors='raise')
data['year'] = data_date.dt.year
data['month'] = data_date.dt.month
data['day'] = data_date.dt.day
plt.figure(figsize=(15,5))
sns.countplot(x="year",data = data)
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(x="month",data = data)
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(x="day",data = data)
plt.show()
import plotly.express as px
 
shootout_by_armed = data['armed'].value_counts()[:94]
shootout_by_armed = pd.DataFrame(shootout_by_armed)
shootout_by_armed =shootout_by_armed.reset_index()#to arrange in descending order
fig = px.pie(shootout_by_armed, values='armed', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()