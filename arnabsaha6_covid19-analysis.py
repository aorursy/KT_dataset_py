# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 

import numpy as np

import os 

import pandas as pd 



import plotly.express as px

import datetime

import seaborn as sns

import plotly.graph_objects as go

import warnings

warnings.filterwarnings('ignore')

import folium 

from folium import plugins
df1 = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
df1.head()
covid= pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

covid.head()
covid= covid.drop(['SNo'],axis=1)
covid['Province/State'] = covid['Province/State'].fillna('Unknown Location',axis=0)

covid.info()
covid_confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

covid_recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

covid_deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
covid['ObservationDate']=pd.to_datetime(covid['ObservationDate'])

covid['Last Update']=pd.to_datetime(covid['Last Update'])
grouping = covid.groupby('ObservationDate')['Last Update', 'Confirmed', 'Deaths'].sum().reset_index()
grouping.head()
fig = px.line(grouping, x="ObservationDate", y="Confirmed", 

              title="Worldwide Confirmed Cases Over Time")

fig.show()
fig = px.line(grouping, x="ObservationDate", y="Deaths", title="Worldwide Deaths Over Time")

fig.show()
china_info = covid[covid['Country/Region'] == "Mainland China"].reset_index()

grouped_china_date = china_info.groupby('ObservationDate')['ObservationDate', 'Confirmed', 'Deaths'].sum().reset_index()
fig = px.line(grouped_china_date, x="ObservationDate", y="Confirmed", 

              title="Confirmed Cases Over Time (MAINLAND CHINA)")

fig.show()
india_info = covid[covid['Country/Region'] == "India"].reset_index()

grouped_india_date = india_info.groupby('ObservationDate')['ObservationDate', 'Confirmed', 'Deaths'].sum().reset_index()
fig = px.line(grouped_india_date, x="ObservationDate", y="Confirmed", 

              title="Confirmed Cases Over Time (REPUBLIC OF INDIA)")

fig.show()
covid19_new = covid

covid19_new['Active'] = covid19_new['Confirmed'] - (covid19_new['Deaths'] + covid19_new['Recovered'])
line_data = covid19_new.groupby(['ObservationDate'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

line_data = line_data.melt(id_vars="ObservationDate", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')
fig = px.line(line_data, x='ObservationDate', y='Count', color='Case', title='Whole World Cases over time')

fig.show()
china_data = covid19_new[covid19_new['Country/Region'] == 'Mainland China'].reset_index(drop=True)

china_line_data = china_data.groupby(['ObservationDate'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

china_line_data = china_line_data.melt(id_vars="ObservationDate", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')
fig = px.line(china_line_data, x='ObservationDate', y='Count',color="Case", title='China Cases over time')

fig.show()
italy_data = covid19_new[covid19_new['Country/Region'] == 'Italy'].reset_index(drop=True)

italy_line_data = italy_data.groupby(['ObservationDate'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

italy_line_data = italy_line_data.melt(id_vars="ObservationDate", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')
fig = px.line(italy_line_data, x='ObservationDate', y='Count',color="Case", title='Italy Cases over time')

fig.show()
fig = px.scatter(covid, y="Deaths",x = "Recovered", color="Country/Region",

                 size='Confirmed')

fig.show()
fig = px.scatter(covid, y="Deaths",x = "Recovered", color="Country/Region",

                 size='Confirmed',log_y=True, log_x=True)

fig.show()
data=df1



data['Date']=pd.to_datetime(data.Date,dayfirst=True)

data['confirmed']=data.ConfirmedForeignNational+data.ConfirmedIndianNational
data.head()
data=data.rename(columns={'Date':'date',

                     'State/UnionTerritory':'state',

                         'Deaths':'deaths'})
latest = data[data['date'] == max(data['date'])].reset_index()

latest_grouped = latest.groupby('state')['confirmed', 'deaths'].sum().reset_index()

latest = data[data['date'] == max(data['date'])]

latest = latest.groupby('state')['confirmed', 'deaths'].max().reset_index()
latest.sort_values('confirmed')
sns.barplot(x='confirmed', y='state',  data=latest.sort_values('confirmed')) 

plt.show()
df= pd.read_csv('/kaggle/input/coronavirus-cases-in-india/Covid cases in India.csv')

India_coord = pd.read_csv('/kaggle/input/coronavirus-cases-in-india/Indian Coordinates.csv')
df.head()
df.drop(['S. No.'],axis=1,inplace=True)
df['Total cases'] = df['Total Confirmed cases (Indian National)'] + df['Total Confirmed cases ( Foreign National )']

df['Active cases'] = df['Total cases'] - (df['Cured/Discharged/Migrated'] + df['Deaths'])

print(f'Number of Confirmed COVID 19 Cases in India', df['Total cases'].sum())

print(f'Number of Active COVID 2019 cases in India:', df['Active cases'].sum())

sns.barplot(x='Total Confirmed cases (Indian National)', y='Name of State / UT',  data=df) 

plt.show()
sns.barplot(x='Total Confirmed cases ( Foreign National )', y='Name of State / UT',  data=df) 

plt.show()
India_coord.drop('Unnamed: 3',axis=1,inplace=True)
India_coord.head(10)
map_data = pd.merge(India_coord,df,on='Name of State / UT')

map_data
map = folium.Map(location=[18, 85], zoom_start=5)



for lat, lon, value, name in zip(map_data['Latitude'], map_data['Longitude'], map_data['Active cases'], map_data['Name of State / UT']):

    folium.CircleMarker([lat, lon],

                        radius=value*.5,

                        color='blue',

                        fill_color='blue',

                        fill_opacity=0.5 ).add_to(map)

map