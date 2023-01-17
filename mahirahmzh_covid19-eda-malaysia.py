# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/uncover/ECDC'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/uncover/ECDC/current-data-on-the-geographic-distribution-of-covid-19-cases-worldwide.csv')
df.head()
is_my = df['countriesandterritories'] =='Malaysia'

df_my = df[is_my]
df_my.head()
print(df_my.shape)
print(df_my.columns)
print(df_my.info())
df.drop(columns=['daterep'])
import datetime





date=df_my.apply(lambda x: datetime.date(int(x['year']), x['month'], x['day']),axis=1)

date = pd.to_datetime(date)

df_my.insert(0, 'date', date)
df_my.head()
import matplotlib.pyplot as plt



plt.figure(figsize=(16,14))



plt.plot(df_my['date'].values, df_my['cases'], lw=2)

plt.xlabel('Dates',)

plt.ylabel('Amount of Cases')





plt.title('Malaysia COVID-19 Number of Cases over Time')

plt.show()
plt.figure(2, figsize=(12,10))

plt.plot(df_my['date'].values, df_my['deaths'], lw=2)

plt.xlabel('Dates',)

plt.ylabel('Amount of Deaths')



plt.title('Malaysia Covid-19 Amount of Deaths Over Time')



plt.show()
new_list = ['Cases', 'Deaths']



plt.figure(3, figsize=(12,10))

plt.plot(df_my['date'].values, df_my['cases'], lw=2)

plt.plot(df_my['date'].values, df_my['deaths'], lw=2)

plt.xlabel('Dates',)

plt.ylabel('Amount of Cases/Deaths')



plt.legend(new_list)

plt.title('Malaysia COVID-19 Amount of Cases vs Deaths')



plt.show()