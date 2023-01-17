# importing package

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 50) # set displaying option



# setting environment

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# read the dataset by using "read_csv" function in pandas

df_performance_daily = pd.read_csv('../input/cee498-project-exploratory-data-analysis-dataset/OperationPerformance_PerRoute-Day-SystemwideOps_8_2-15_2020.csv')

df_performance_total = pd.read_csv('../input/cee498-project-exploratory-data-analysis-dataset/OperationPerformance_PerRoute-SystemwideOps_8_2-15_2020.csv')

df_ridership = pd.read_csv('../input/cee498-project-exploratory-data-analysis-dataset/Ridershippertrip_PerTrip-AugustWeekdays9-9.csv')
df_performance_daily.head()
df_performance_total.head()
df_ridership.head()
df_performance_daily = df_performance_daily.drop(['EMPTY_1', 'EMPTY_2'], axis = 1)
df_performance_total = df_performance_total.drop(['EMPTY_1', 'EMPTY_2', 'EMPTY_3'], axis = 1)
df_ridership = df_ridership.drop(['Duty', 'EMPTY_1', 'EMPTY_2', 'EMPTY_3', 'Graphic'], axis = 1)
# eliminate zero-value column

df_ridership = df_ridership.drop(['Capacity', 'Full capacity', 'Capacity (pract.)', 'Load factor [%]', 'Load factor (pract.)[%]', 'PM factor [%]'], axis = 1)



# eliminate string-value column

df_ridership = df_ridership.drop(['Veh. type', 'Pattern'], axis = 1)
plt.scatter(df_ridership['No'],df_ridership['Load avg'])

plt.xlabel('No')

plt.ylabel('Load Avg')

plt.show()
df_ridership['Load avg'].describe()
df_ridership.iloc[0:700].groupby('Line')['Load avg'].median().plot.bar(color="#9494b8")

plt.xlabel('Bus Line')

plt.ylabel('Load Avg')

plt.title('Line Number -> Load Avg')

plt.show()
df_ridership.groupby('Date')['Load avg'].median().plot.bar()

plt.xlabel('Bus Line')

plt.ylabel('Load Avg')

plt.title('Date -> Load Avg')

plt.show()
list = ['BLUE[V:0001]', 'BROWN[V:0001]', 'GOLD', 'GREEN[V:0001]']

num = 1

plt.figure(figsize=(100, 380))

for item in list:

    plt.subplot(22,2,num)

    sns.lineplot(data=df_ridership.loc[df_ridership['Line'] == item], x="Sched. start", y="Load avg")

    num = num + 1
plt.figure(figsize=(10, 5))

df_ridership.groupby('P-Stops')['Load avg'].median().plot.bar()

plt.xlabel('P-stops')

plt.ylabel('Load Avg')

plt.title('P-stops -> Load Avg')

plt.show()
plt.figure(figsize=(20, 5))

df_ridership.groupby('M-Stops')['Load avg'].median().plot.bar()

plt.xlabel('M-Stops')

plt.ylabel('Load Avg')

plt.title('M-Stops -> Load Avg')

plt.show()
list = ['BLUE[V:0001]', 'BROWN[V:0001]', 'GOLD', 'GREEN[V:0001]']

num = 1

plt.figure(figsize=(100, 380))

for item in list:

    plt.subplot(22,2,num)

    sns.boxplot(data=df_ridership.loc[df_ridership['Line'] == item], x="Sched. start", y="Load avg")

    num = num + 1
plt.figure(figsize=(10, 5))

sns.boxplot(data=df_ridership, x="P-Stops", y="Load avg")