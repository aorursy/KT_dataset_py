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
df_1 = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv', delimiter=',', parse_dates=['diagnosed_date', 'status_change_date'])

df_2 = pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv', delimiter=',', parse_dates=['Date'])

df_3 = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv', delimiter=',')

df_4 = pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingDetails.csv', delimiter=',')

df_5 = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv', delimiter=',')

df_6 = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv', delimiter=',')

df_7 = pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingLabs.csv', delimiter=',')

df_8 = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv', delimiter=',')

print('IndividualDetails dataset shape: {}\n'.format(df_1.shape))

df_1.info()
df_1.drop(['id'], axis = 1, inplace = True)

df_1 = df_1.drop_duplicates()

df_1 = df_1.reset_index(drop = True)
df_1.head(5)
plt.figure(figsize = (15, 7))

sns.heatmap(df_1.isnull())
df_1.loc[df_1['age'] == '28-35', 'age'] = 31.5
df_1['diagnosed_month'] = df_1['diagnosed_date'].apply(lambda x: str(x).split('-')[1])
df_1['detected_state'].value_counts().plot(kind = 'bar', figsize = (15, 7))

plt.legend()
df_1.loc[df_1['detected_state'] == 'Tamil Nadu']['diagnosed_date'].value_counts().plot(kind = 'bar', figsize = (15, 7))

plt.legend()
df_1['diagnosed_date'].value_counts().plot(kind = 'bar', figsize = (15, 7))

plt.legend()
df_1['diagnosed_month'].value_counts().plot(kind = 'bar', figsize = (15, 7))

plt.legend()
df_1['nationality'].value_counts().plot(kind = 'bar', figsize = (15, 7))

plt.legend()
plt.figure(figsize = (15, 7))

sns.kdeplot(df_1['age'], shade = True)
#df_2.drop(['id'], axis = 1, inplace = True)

df_2 = df_2.drop_duplicates()

df_2 = df_2.reset_index(drop = True)
print('StatewiseTestingDetails dataset shape: {}\n'.format(df_2.shape))

df_2.info()
df_2.head()
df = df_2.groupby('State').agg({'TotalSamples' : ['sum'], 'Negative':  ['sum'], 'Positive' : ['sum']})

df.columns = ['TotalSamples', 'Negative', 'Positive']

df.plot(kind = 'bar', figsize = (15, 7))

plt.legend()