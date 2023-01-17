import numpy as np 

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

from learntools.geospatial.tools import geocode

%matplotlib inline

plt.style.use('ggplot')

sns.set_style('dark')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv',parse_dates=True)

test_df=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv',parse_dates=True)
train_df.tail()
test_df.head()
print('train datarame shape: ',train_df.shape)

print('test datarame shape: ',test_df.shape)
train_df.info()
test_df.info()
pattern1='[0-9]{4}-([0-9]{2})-[0-9]{2}'

pattern2='[0-9]{4}-[0-9]{2}-([0-9]{2})'

train_df['month']=train_df['Date'].str.extract(pattern1).astype(int)

test_df['month']=test_df['Date'].str.extract(pattern1).astype(int)

train_df['day']=train_df['Date'].str.extract(pattern2).astype(int)

test_df['day']=test_df['Date'].str.extract(pattern2).astype(int)
confirmed_cases=train_df.pivot_table(values='ConfirmedCases',index='Date',columns='Country/Region')
countries=[]

for i in train_df['Country/Region'].unique():

    val=confirmed_cases.loc['2020-03-22',i]

    if val>1000:

        countries.append(i)

print(len(countries))
fig=plt.figure(figsize=(24,24))

for i,j in zip(countries,range(20)):

    ax=fig.add_subplot(5,4,j+1)

    ax.plot(confirmed_cases.index,confirmed_cases[i])

    ax.set_xticklabels(confirmed_cases.index,rotation=90)

    ax.set_title('confirmed cases in {} from 22-1 to 22-3'.format(i))

    plt.subplots_adjust(hspace=0.8)

plt.show()
confirmed_cases2=train_df.pivot_table(values='ConfirmedCases',columns='Date',index='Country/Region').sort_values('2020-03-22',ascending=False).reset_index()
top_10=confirmed_cases2.head(10)

g=sns.catplot(x='Country/Region',

            y='2020-03-22',

            data=top_10,

            kind='bar')

g.fig.suptitle('confirmed cases in the top 10 countries in 22-03-2020',y=1.05)

g.set(xlabel='Country',ylabel='Confirmed Cases till 22-03-2020')

plt.xticks(rotation=90)

plt.show()
safe_countries=confirmed_cases2.tail(10)

g=sns.catplot(x='Country/Region',

            y='2020-03-22',

            data=safe_countries,

            kind='bar')

g.fig.suptitle('the most safe countries in 2020-03-22',y=1.05)

plt.xticks(rotation=90)

plt.show()
fig, ax=plt.subplots(figsize=(20,5))

ax.plot(confirmed_cases.index,confirmed_cases['Egypt'],color='black')

ax.set_xticklabels(confirmed_cases.index,rotation=90)

ax.set_title('confirmed cases in Egypt from 22-1 to 22-3')

plt.show()
deaths=train_df.pivot_table(values='Fatalities',index='Date',columns='Country/Region')
countries=[]

for i in train_df['Country/Region'].unique():

    val=deaths.loc['2020-03-22',i]

    if val>100:

        countries.append(i)

print(len(countries))
fig=plt.figure(figsize=(15,15))

for i,j in zip(countries,range(4)):

    ax=fig.add_subplot(2,2,j+1)

    ax.plot(deaths.index,deaths[i])

    ax.set_xticklabels(deaths.index,rotation=90)

    ax.set_title('Fatalities in {} from 22-1 to 22-3'.format(i))

    plt.subplots_adjust(hspace=0.7)

plt.show()
deaths2=train_df.pivot_table(values='Fatalities',columns='Date',index='Country/Region').sort_values('2020-03-22',ascending=False).reset_index()
top_10=deaths2.head(10)

g=sns.catplot(x='Country/Region',

            y='2020-03-22',

            data=top_10,

            kind='bar')

g.fig.suptitle('Fatalities in the top 10 countries in 22-03-2020',y=1.05)

g.set(xlabel='Country',ylabel='Fatalities till 22-03-2020')

plt.xticks(rotation=90)

plt.show()