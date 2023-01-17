# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv')

df_new = df.copy()
df_new.head()
df_new = df_new.rename(columns={'Cumulative number of case(s)':'Confirmed cases', \

                        'Number of deaths': 'Deaths',\

                       'Number recovered': 'Recovered cases'})
df_new.info()
df_new['Country'].unique()
plt.figure(figsize=(10,10))

sns.barplot(y = df_new['Country'] , x = df_new['Confirmed cases'])

plt.xticks(rotation = 90)
df1 = df_new[df_new['Country'].str.contains("China") ]

df1
plt.figure(figsize=(10,5))

sns.countplot('Country',data=df1)
df2 = df1[df1['Country'] == 'China']

plt.figure(figsize=(20,5))

sns.barplot(x = df2['Date'] , y = df2['Confirmed cases'])

plt.title('Cases in China per Date')

plt.xticks(rotation = 90)
df2 = df1[df1['Country'] == 'China']

plt.figure(figsize=(20,5))

sns.barplot(x = df2['Date'] , y = df2['Deaths'])

plt.title('Deaths  in China per Date')

plt.xticks(rotation = 90)
by_date = df_new.groupby('Date')['Confirmed cases', 'Deaths', 'Recovered cases'].sum().reset_index()

by_date
df_melt_bydate = by_date.melt(id_vars='Date', value_vars=['Confirmed cases', 'Deaths', 'Recovered cases'])

df_melt_bydate
plt.figure(figsize=(20,10))

sns.lineplot(x = 'Date' , y = 'value',data = df_melt_bydate  , hue = 'variable')

plt.xticks(rotation = 90)

plt.annotate(xy = ("2003-05-29",8295), s ='The curve starts to flatten',xytext=(55,8295))

plt.title('Worldwide confirmed and recovered cases, and deaths over time')

plt.show()

df_new['Date'] = pd.to_datetime(df_new['Date'])

df_new.insert(1,'Week',df_new['Date'].dt.week)
recover_by_week = df_new.groupby('Week')['Recovered cases'].sum().reset_index()
plt.figure(figsize=(20,10))

plt.title('Recovered Cases by week worldwide')

sns.lineplot(x ='Week' , y = 'Recovered cases' ,data= recover_by_week)