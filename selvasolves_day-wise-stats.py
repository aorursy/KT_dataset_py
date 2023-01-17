# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import datetime



plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

sns.set_style("dark")

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Load the data and a quick view

def adjustformat(x):

    if '1/31/2020' in x:

        return '2020-31-01 ' + x.split()[1] + ":00"

    else:

        return x

    

corona_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

corona_df1 = corona_df[:497]

corona_df1['Date'] = pd.to_datetime(corona_df1['Date'])

corona_df2 = corona_df[497:]

corona_df2['Date'] = corona_df2['Date'].apply(lambda x: adjustformat(x))

corona_df2['Date'] = pd.to_datetime(corona_df2['Date'],format = '%Y-%d-%m %H:%M:%S')

# corona_df2.head()

corona_df = corona_df1.append(corona_df2)

corona_df.head()
#convert the Last update column to datetime

corona_df['Date'] = pd.to_datetime(corona_df['Date'])
#Quick check on data distribution

corona_df.info()
#Lets do one more check

corona_df[(corona_df.Confirmed == 0) & (corona_df.Deaths == 0) & (corona_df.Recovered ==0)]
#Lets add day count to look at the trend

start_date = corona_df.sort_values(['Date'], ascending=[True]).head(1).values[0][1]

print(start_date)

#start_date = corona_df['Last Update'].min()

corona_df['nthday'] = corona_df['Date'].apply(lambda x: ((x - start_date).days)+1)

current_day = corona_df['nthday'].max()
#Checking how to take the counts.

corona_df['Confirmed'].sum()
corona_df[corona_df.nthday == current_day]['Confirmed'].sum()
#Lets quickly plot the trend.

corona_df.groupby(['nthday'])['Confirmed'].sum().plot()
#death rate vs recovery rate

dr_rate = corona_df.groupby(['nthday'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

dr_rate['Deathrate'] = (dr_rate['Deaths'] / dr_rate['Confirmed'])*100

dr_rate['Recoveryrate'] = (dr_rate['Recovered'] / dr_rate['Confirmed'])*100



sns.lineplot(x='nthday', y='Deathrate', data=dr_rate, label="Death", color='red')

sns.lineplot(x='nthday', y='Recoveryrate', data=dr_rate, label="Recovery", color='green')
#No of countries affected so far

corona_df[corona_df.nthday == current_day].groupby(['Country'])['Confirmed', 'Deaths','Recovered' ].sum().sort_values(['Confirmed'], ascending=[False])
#Lets see the country spread by the day

corona_df.groupby(['Country']).agg({'nthday' : 'min'}).reset_index().sort_values(['nthday'])
#

corona_df.groupby(['nthday'])['Country'].nunique().plot()
corona_df[(corona_df.nthday == current_day) & (corona_df.Country.isin(['China', 'Mainland China']))].groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index().sort_values(['Confirmed'], ascending=[False])