# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "seaborn"

from plotly.subplots import make_subplots



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

df = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv", parse_dates = ['Date'])

df.head()
df.info()
df.rename(columns={'Province/State':'State','Country/Region':'Country'},inplace=True)

df.describe(include='object')
date=df.Date.value_counts().sort_index()

print('First date:',date.index[0])

print('Last date:',date.index[-1])
df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
df.describe()
df[df['Active']==df['Active'].min()]
df.loc[(df['Active']==-6),'Recovered']=162

# Let us once again find active cases

df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
df.describe()
df[df['Active']==df['Active'].min()]
df.loc[(df['Active']==-1),'Confirmed']=1

# Let us once again find active cases

df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
df.describe()
latest=df[df['Date']==df['Date'].max()]

active_country=latest.groupby('Country')['Confirmed','Deaths','Recovered','Active'].sum().reset_index()

active_country.sort_values(by='Active',ascending=False,inplace=True)

active_country.head()
active_top_20=active_country.head(20)
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(15,20))

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Active Cases',fontsize=25)

plt.ylabel('Countries',fontsize=25)

plt.title('Top 20 Countries with Active Cases',fontsize=50)

ax=sns.barplot(x=active_top_20['Active'],y=active_top_20['Country'])

for i, (value, name) in enumerate(zip(active_top_20['Active'], active_top_20['Country'])):

    ax.text(value, i-.05, f'{value:,.0f}',  size=10, ha='left',  va='center')

ax.set(xlabel='Active Cases',ylabel='Countries')
active_country.sort_values('Confirmed',ascending=False,inplace=True)

con_top_20=active_country.head(20)

con_top_20.head()
plt.figure(figsize= (15,10))

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)

plt.xlabel("Total Confirmed cases",fontsize = 25)

plt.ylabel('Country',fontsize = 25)

plt.title("Top 20 countries with Confirmed cases" , fontsize = 50)

ax = sns.barplot(x = con_top_20.Confirmed, y = con_top_20.Country)

for i, (value, name) in enumerate(zip(con_top_20.Confirmed,con_top_20.Country)):

    ax.text(value, i-.05, f'{value:,.0f}',  size=10, ha='left',  va='center')

ax.set(xlabel='Confirmed cases', ylabel='Country')
active_country.sort_values('Deaths',ascending=False,inplace=True)

death_top_20=active_country.head(20)

death_top_20.head()
plt.figure(figsize= (15,10))

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)

plt.xlabel("Total cases",fontsize = 25)

plt.ylabel('Country',fontsize = 25)

plt.title("Top 20 countries with Deaths" , fontsize = 50)

ax = sns.barplot(x = death_top_20.Deaths, y = death_top_20.Country)

for i, (value, name) in enumerate(zip(death_top_20.Deaths,death_top_20.Country)):

    ax.text(value, i-.05, f'{value:,.0f}',  size=10, ha='left',  va='center')

ax.set(xlabel='Total Deaths', ylabel='Country')
active_country.sort_values('Recovered',ascending=False,inplace=True)

rec_top_20=active_country.head(20)

rec_top_20.head()
plt.figure(figsize= (15,10))

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)

plt.xlabel("Total Recovered",fontsize = 25)

plt.ylabel('Country',fontsize = 25)

plt.title("Top 20 countries with Recoveries" , fontsize = 50)

ax = sns.barplot(x = rec_top_20.Recovered, y = rec_top_20.Country)

for i, (value, name) in enumerate(zip(rec_top_20.Recovered,rec_top_20.Country)):

    ax.text(value, i-.05, f'{value:,.0f}',  size=10, ha='left',  va='center')

ax.set(xlabel='Total Recovered', ylabel='Country')
ch_df=df[df['Country']=='China']

ch_df.head()
plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 15)

plt.xlabel("Dates",fontsize = 30)

plt.ylabel('Total cases',fontsize = 30)

plt.title("Confirmed Cases in China Over Time" , fontsize = 30)

total_cases_ch = ch_df.groupby('Date')['Date', 'Confirmed'].sum().reset_index()

total_cases_ch['Date'] = pd.to_datetime(total_cases_ch['Date'])





ax = sns.pointplot( x = total_cases_ch.Date.dt.date ,y = total_cases_ch.Confirmed , color = 'r')

ax.set(xlabel='Dates', ylabel='Total Confirmed cases')
it_df=df[df['Country']=='Italy']

it_df.head()
plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 15)

plt.xlabel("Dates",fontsize = 30)

plt.ylabel('Total cases',fontsize = 30)

plt.title("Confirmed Cases in Italy Over Time" , fontsize = 30)

total_cases_it = it_df.groupby('Date')['Date', 'Confirmed'].sum().reset_index()

total_cases_it['Date'] = pd.to_datetime(total_cases_it['Date'])





ax = sns.pointplot( x = total_cases_it.Date.dt.date ,y = total_cases_it.Confirmed , color = 'b')

ax.set(xlabel='Dates', ylabel='Total Confirmed cases')
us_df=df[df['Country']=='US']

us_df.head()
plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 15)

plt.xlabel("Dates",fontsize = 30)

plt.ylabel('Total cases',fontsize = 30)

plt.title("Confirmed Cases in US Over Time" , fontsize = 30)

total_cases_us = us_df.groupby('Date')['Date', 'Confirmed'].sum().reset_index()

total_cases_us['Date'] = pd.to_datetime(total_cases_it['Date'])





ax = sns.pointplot( x = total_cases_us.Date.dt.date ,y = total_cases_us.Confirmed , color = 'g')

ax.set(xlabel='Dates', ylabel='Total Confirmed cases')
rate_df = latest.groupby(by = 'Country')['Recovered','Confirmed','Deaths'].sum().reset_index()

rate_df['Recovery percentage'] =  round(((rate_df['Recovered']) / (rate_df['Confirmed'])) * 100 , 2)

rate_df['Death percentage'] =  round(((rate_df['Deaths']) / (rate_df['Confirmed'])) * 100 , 2)

rate_df.head()
rate_df.sort_values(by='Death percentage',ascending=False,inplace=True)

mortal_df=rate_df.head(20)

mortal_df.head()
plt.figure(figsize= (20,20))

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)

plt.xlabel("Total cases",fontsize = 30)

plt.ylabel('Country',fontsize = 30)

plt.title("Top 20 countries having highest mortality rate" , fontsize = 30)

ax = sns.barplot(x = mortal_df['Death percentage'], y = mortal_df['Country'])

for i, (value, name) in enumerate(zip(mortal_df['Death percentage'], mortal_df['Country'])):

    ax.text(value, i-.05, f'{value:,.0f}',  size=10, ha='left',  va='center')

ax.set(xlabel='Mortality Rate in percentage', ylabel='Country')
rate_df.sort_values(by='Recovery percentage',ascending=False,inplace=True)

rec_df=rate_df.head(20)

rec_df.head()
plt.figure(figsize= (20,20))

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)

plt.xlabel("Total cases",fontsize = 30)

plt.ylabel('Country',fontsize = 30)

plt.title("Top 20 countries having highest Recovery rate" , fontsize = 30)

ax = sns.barplot(x = rec_df['Recovery percentage'], y = rec_df['Country'])

for i, (value, name) in enumerate(zip(rec_df['Recovery percentage'], rec_df['Country'])):

    ax.text(value, i-.05, f'{value:,.0f}',  size=10, ha='left',  va='center')

ax.set(xlabel='Recovery Rate in percentage', ylabel='Country')
df1 = df

df1['Date'] = pd.to_datetime(df1['Date'])

df1['Date'] = df1['Date'].dt.strftime('%m/%d/%Y')

df1 = df1.fillna('-')

fig = px.density_mapbox(df1, lat='Lat', lon='Long', z='Confirmed', radius=20,zoom=1, hover_data=["Country",'State',"Confirmed"],

                        mapbox_style="carto-positron", animation_frame = 'Date', range_color= [0, 1000],title='Spread of Covid-19')

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

fig.show()
figure = px.choropleth(active_country, locations="Country", 

                    locationmode='country names', color="Active", 

                    hover_name="Country", range_color=[1,1000], 

                    color_continuous_scale="blues", 

                    title='Countries with Active Cases')

figure.show()
china=df[df['Country']=='China']

china.head()
ch=china.groupby('State')['Confirmed','Deaths','Recovered','Active'].sum().reset_index()

ch.head()
plt.figure(figsize=(20,20))

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('No. of Confirmed Cases in China',fontsize=30)

plt.ylabel('States',fontsize=30)

plt.title('States having Active Cases in China', fontsize=30)

ax=sns.barplot(x=ch['Confirmed'],y=ch['State'])

for i, (value, name) in enumerate(zip(ch.Confirmed, ch.State)):

    ax.text(value, i-.05, f'{value:,.0f}',  size=10, ha='left',  va='center')

ax.set(xlabel='Total cases', ylabel='States')
ind_df=df[df['Country']=='India']

ind_df.head()
plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 15)

plt.xlabel("Dates",fontsize = 30)

plt.ylabel('Total cases',fontsize = 30)

plt.title("Confirmed Cases in India Over Time" , fontsize = 30)

total_cases_ind = ind_df.groupby('Date')['Date', 'Confirmed'].sum().reset_index()

total_cases_ind['Date'] = pd.to_datetime(total_cases_ind['Date'])





ax = sns.pointplot( x = total_cases_ind.Date.dt.date ,y = total_cases_ind.Confirmed , color = 'y')

ax.set(xlabel='Dates', ylabel='Total Confirmed cases')
plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 15)

plt.xlabel("Dates",fontsize = 30)

plt.ylabel('Total deaths',fontsize = 30)

plt.title("Deaths in India Over Time" , fontsize = 30)

total_deaths_ind = ind_df.groupby('Date')['Date', 'Deaths'].sum().reset_index()

total_deaths_ind['Date'] = pd.to_datetime(total_deaths_ind['Date'])





ax = sns.pointplot( x = total_deaths_ind.Date.dt.date ,y = total_deaths_ind.Deaths , color = 'y')

ax.set(xlabel='Dates', ylabel='Total Deaths')
plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 15)

plt.xlabel("Dates",fontsize = 30)

plt.ylabel('Total Recoveries',fontsize = 30)

plt.title("Recoveries in India Over Time" , fontsize = 30)

total_rec_ind = ind_df.groupby('Date')['Date', 'Recovered'].sum().reset_index()

total_rec_ind['Date'] = pd.to_datetime(total_rec_ind['Date'])





ax = sns.pointplot( x = total_rec_ind.Date.dt.date ,y = total_rec_ind.Recovered , color = 'y')

ax.set(xlabel='Dates', ylabel='Total Recoveries')