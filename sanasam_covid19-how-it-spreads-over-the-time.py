# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

from plotly.subplots import make_subplots

from datetime import datetime
df_clean=pd.read_csv('../input/covid19-clean-data/COVID19_clean_data.csv')

df_clean.head()
df_clean.info()
df_clean.describe()
df_clean[['Province/State']]=df_clean[['Province/State']].fillna(value='')

df_clean.head()
df_clean.info()
df_clean['Under_treatment']=df_clean['Confirmed']-df_clean['Deaths']-df_clean['Recovered']

df_clean.head()
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
print('First entry',df_clean[['Date']].min())

print('Last  entry',df_clean[['Date']].max())

print('Total time period',df_clean[['Date']].max()-df_clean[['Date']].min())
df_clean_date=df_clean.groupby('Date')['Confirmed','Deaths','Recovered','Under_treatment'].sum().reset_index()

print('First 5 rows',df_clean_date.head())

print('***'*100)

print('Last 5 rows',df_clean_date.tail())
sns.set_style("darkgrid")

plt.figure(figsize=(8,4))

sns.lineplot(x='Date',y='Confirmed',data=df_clean_date)

plt.show

sns.lineplot(x='Date',y='Deaths',data=df_clean_date)

plt.ylabel('No. of cases')

plt.legend(['No. of confirm case over time'])

plt.title(' NO. of COnfirm case Vs. no of Deaths Over time')



plt.xticks(rotation=60)

plt.show()









sns.set_style("darkgrid")

plt.figure(figsize=(8,4))

sns.lineplot(x='Date',y='Confirmed',data=df_clean_date)

sns.lineplot(x='Date',y='Recovered',data=df_clean_date)

plt.ylabel('No. of cases')

plt.legend(['No. of confirm case over time'])

plt.title(' NO. of COnfirm case Vs. no of Recovers Over time')



plt.xticks(rotation=60)



plt.show()









sns.set_style("darkgrid")

plt.figure(figsize=(8,4))

sns.lineplot(x='Date',y='Confirmed',data=df_clean_date)

sns.lineplot(x='Date',y='Under_treatment',data=df_clean_date)

plt.ylabel('No. of cases')

plt.legend(['No. of confirm case over time'])

plt.title(' NO. of COnfirm case Vs. no of Under_treatment Over time')



plt.xticks(rotation=60)

plt.show()





df_clean_country=df_clean.groupby('Country/Region')['Confirmed','Deaths','Recovered','Under_treatment'].sum().reset_index().sort_values('Confirmed',ascending=False)

df_clean_country.head()
top_10=df_clean_country[:10]

top_10
plt.bar(top_10['Country/Region'],top_10['Confirmed'])

plt.bar(top_10['Country/Region'],top_10['Deaths'])

plt.ylabel('No.of cases')

plt.title('Top 10 infected countries: Cconfirm cases Vs Deaths')

plt.xticks(rotation=65)

plt.show





plt.bar(top_10['Country/Region'],top_10['Confirmed'])

plt.bar(top_10['Country/Region'],top_10['Recovered'])

plt.ylabel('No.of cases')

plt.title('Top 10 infected countries: Cconfirm cases Vs Recovered')

plt.xticks(rotation=65)

plt.show





plt.bar(top_10['Country/Region'],top_10['Confirmed'])

plt.bar(top_10['Country/Region'],top_10['Under_treatment'])

plt.ylabel('No.of cases')

plt.title('Top 10 infected countries: Cconfirm cases Vs Under treatment')

plt.xticks(rotation=65)

plt.show
df_clean_country['Recovery rate %']=df_clean_country['Recovered']/df_clean_country['Confirmed']*100

df_clean_country['Death rate %']=df_clean_country['Deaths']/df_clean_country['Confirmed']*100

df_clean_country.head()
top_10['Death rate %']=top_10['Deaths']/top_10['Confirmed']*100

top_10['Recovery rate %']=top_10['Recovered']/top_10['Confirmed']*100

top_10.head()
sns.barplot(y='Country/Region',x='Death rate %',data=top_10)

plt.title(' Top 10 Infected country Deaths rate  in %')
sns.barplot(y='Country/Region',x='Recovery rate %',data=top_10)

plt.title(' Top 10 Infected country Recovery rate  in %')
df_spread=df_clean.groupby(['Date','Country/Region'])['Confirmed','Recovered'].sum().reset_index()

df_spread.head()
df_spread['Confirmed_scale']=df_spread['Confirmed'].pow(0.45)

df_spread.head()


df_spread['Date'] = pd.to_datetime(df_spread['Date'])

df_spread['Date']=df_spread['Date'].dt.strftime('%d%m%Y')

df_spread.head()
map_spread=px.scatter_geo(df_spread, locations='Country/Region',locationmode='country names',color='Confirmed',

                         size='Confirmed_scale',hover_name='Country/Region', range_color=[0,20000],

                          projection='natural earth', animation_frame='Date',color_continuous_scale="portland",title='Virus: How virus spread worldwide')

map_spread.show()
map_recovered=px.scatter_geo(df_spread, locations='Country/Region',locationmode='country names',color='Recovered',

                         size='Confirmed_scale',hover_name='Country/Region', range_color=[0,20000],

                          projection='natural earth', animation_frame='Date',color_continuous_scale="portland",title='Virus: How virus spread worldwide')

map_recovered.show()