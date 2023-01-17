# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reading data from CSV file



df = pd.read_csv(r'/kaggle/input/us-accidents/US_Accidents_Dec19.csv')
# Summarizing the Data



df.describe()
# Seeing all the columns



df.columns
plt.figure(figsize=(16,10))

states=df['State'].unique()

count_by_state=list(df['State'].value_counts())

sns.barplot(x=states,y=count_by_state)
plt.figure(figsize=(16,10))

severity=df['Severity'].unique()

count_by_severity=list(df['Severity'].value_counts())

sns.barplot(x=severity,y=count_by_severity)
plt.figure(figsize=(16,10))



severity_1_by_state = []

severity_2_by_state = []

severity_3_by_state = []

severity_4_by_state = []



for i in states:

    severity_1_by_state.append(df[(df['Severity']==1)&(df['State']==i)].count()['ID'])

    severity_2_by_state.append(df[(df['Severity']==2)&(df['State']==i)].count()['ID'])

    severity_3_by_state.append(df[(df['Severity']==3)&(df['State']==i)].count()['ID'])

    severity_4_by_state.append(df[(df['Severity']==4)&(df['State']==i)].count()['ID'])



    

plt.bar(states, severity_2_by_state, label='Severity 2')

plt.bar(states, severity_3_by_state, label='Severity 3')

plt.bar(states, severity_4_by_state, label='Severity 4')

plt.bar(states, severity_1_by_state, label='Severity 1')

plt.xlabel('State')

plt.ylabel('Accident Count')

plt.legend()
TMC_counts=df.TMC.value_counts()

plt.figure(figsize=(16, 10))



ax=sns.barplot(TMC_counts.index, TMC_counts)

ax.set(xlabel='TMC type', ylabel='Count')
weather_counts=df['Weather_Condition'].value_counts()

plt.figure(figsize=(16, 10))



ax=sns.barplot(weather_counts.index[:11], weather_counts[:11])

ax.set(xlabel='Weather Condition', ylabel='Accident Count')
severity_1_by_weather = []

severity_2_by_weather = []

severity_3_by_weather = []

severity_4_by_weather = []

weather = df.Weather_Condition.value_counts()

for i in weather.index:

    severity_1_by_weather.append(df[(df['Severity']==1)&(df['Weather_Condition']==i)].count()['ID'])

    severity_2_by_weather.append(df[(df['Severity']==2)&(df['Weather_Condition']==i)].count()['ID'])

    severity_3_by_weather.append(df[(df['Severity']==3)&(df['Weather_Condition']==i)].count()['ID'])

    severity_4_by_weather.append(df[(df['Severity']==4)&(df['Weather_Condition']==i)].count()['ID'])
plt.figure(figsize=(16, 10))

plt.bar(weather.index[:11], severity_2_by_weather[:11], label='Severity 2')

plt.bar(weather.index[:11], severity_3_by_weather[:11], label='Severity 3')

plt.bar(weather.index[:11], severity_4_by_weather[:11], label='Severity 4')

plt.bar(weather.index[:11], severity_1_by_weather[:11], label='Severity 1')

plt.xlabel('Weather Condition')

plt.ylabel('Accident Count')

plt.legend()