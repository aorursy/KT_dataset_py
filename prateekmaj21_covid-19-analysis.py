#importing the various libraries



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
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df1 = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
df1.head()
covid= pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

covid.head()
covid.info()
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
#using plotly



fig = px.line(grouping, x="ObservationDate", y="Confirmed", 

              title="Worldwide Confirmed Cases Over Time")

fig.show()
fig = px.line(grouping, x="ObservationDate", y="Confirmed", 

              title="Worldwide Confirmed Cases (Logarithmic Scale) Over Time", 

              log_y=True)

fig.show()
fig = px.line(grouping, x="ObservationDate", y="Deaths", title="Worldwide Deaths Over Time")

fig.show()
fig = px.line(grouping, x="ObservationDate", y="Deaths", title="Worldwide Deaths (Logarithmic Scale) Over Time", 

              log_y=True)

fig.show()
china_info = covid[covid['Country/Region'] == "Mainland China"].reset_index()

grouped_china_date = china_info.groupby('ObservationDate')['ObservationDate', 'Confirmed', 'Deaths'].sum().reset_index()
fig = px.line(grouped_china_date, x="ObservationDate", y="Confirmed", 

              title="Confirmed Cases Over Time (MAINLAND CHINA)")

fig.show()
fig = px.line(grouped_china_date, x="ObservationDate", y="Confirmed", 

              title="Confirmed Cases Over Time (MAINLAND CHINA)- Logarithmic Graph",log_y=True)

fig.show()
india_info = covid[covid['Country/Region'] == "India"].reset_index()

grouped_india_date = india_info.groupby('ObservationDate')['ObservationDate', 'Confirmed', 'Deaths'].sum().reset_index()
fig = px.line(grouped_india_date, x="ObservationDate", y="Confirmed", 

              title="Confirmed Cases Over Time (REPUBLIC OF INDIA)")

fig.show()
fig = px.line(grouped_india_date, x="ObservationDate", y="Confirmed", 

              title="Confirmed Cases Over Time (REPUBLIC OF INDIA)- Logarithmic Graph",log_y=True)

fig.show()
italy_info = covid[covid['Country/Region'] == "Italy"].reset_index()

grouped_italy_date = italy_info.groupby('ObservationDate')['ObservationDate', 'Confirmed', 'Deaths'].sum().reset_index()
fig = px.line(grouped_italy_date, x="ObservationDate", y="Confirmed", 

              title="Confirmed Cases Over Time (Italy)")

fig.show()
fig = px.line(grouped_italy_date, x="ObservationDate", y="Confirmed", 

              title="Confirmed Cases Over Time (Italy)- Logarithmic", log_y=True)

fig.show()
#spain

spain_info = covid[covid['Country/Region'] == "Spain"].reset_index()

grouped_spain_date = spain_info.groupby('ObservationDate')['ObservationDate', 'Confirmed', 'Deaths'].sum().reset_index()
fig = px.line(grouped_spain_date, x="ObservationDate", y="Confirmed", 

              title="Confirmed Cases Over Time (Spain)")

fig.show()
fig = px.line(grouped_spain_date, x="ObservationDate", y="Confirmed", 

              title="Confirmed Cases Over Time (Spain)- Logarithmic Graph",log_y=True)

fig.show()
#USA



usa_info = covid[covid['Country/Region'] == "US"].reset_index()

grouped_usa_date = usa_info.groupby('ObservationDate')['ObservationDate', 'Confirmed', 'Deaths'].sum().reset_index()
fig = px.line(grouped_usa_date, x="ObservationDate", y="Confirmed", 

              title="Confirmed Cases Over Time (USA)")

fig.show()
fig = px.line(grouped_usa_date, x="ObservationDate", y="Confirmed", 

              title="Confirmed Cases Over Time (USA)- Logarithmic Graph",log_y=True)

fig.show()
#worldwide cases over time



covid19_new = covid

covid19_new['Active'] = covid19_new['Confirmed'] - (covid19_new['Deaths'] + covid19_new['Recovered'])
line_data = covid19_new.groupby(['ObservationDate'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

line_data = line_data.melt(id_vars="ObservationDate", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')
fig = px.line(line_data, x='ObservationDate', y='Count', color='Case', title='Whole World Cases over time')

fig.show()
# cases in china over time



china_data = covid19_new[covid19_new['Country/Region'] == 'Mainland China'].reset_index(drop=True)

china_line_data = china_data.groupby(['ObservationDate'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

china_line_data = china_line_data.melt(id_vars="ObservationDate", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')
fig = px.line(china_line_data, x='ObservationDate', y='Count',color="Case", title='China Cases over time')

fig.show()
# cases in Italy over time



italy_data = covid19_new[covid19_new['Country/Region'] == 'Italy'].reset_index(drop=True)

italy_line_data = italy_data.groupby(['ObservationDate'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

italy_line_data = italy_line_data.melt(id_vars="ObservationDate", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')
fig = px.line(italy_line_data, x='ObservationDate', y='Count',color="Case", title='Italy Cases over time')

fig.show()
#USA





usa_data = covid19_new[covid19_new['Country/Region'] == 'US'].reset_index(drop=True)

usa_line_data = usa_data.groupby(['ObservationDate'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

usa_line_data = usa_line_data.melt(id_vars="ObservationDate", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')
fig = px.line(usa_line_data, x='ObservationDate', y='Count',color="Case", title='USA Cases over time')

fig.show()
covid.head()
fig = px.scatter(covid, y="Deaths",x = "Recovered", color="Country/Region",

                 size='Confirmed')

fig.show()
fig = px.scatter(covid, y="Deaths",x = "Recovered", color="Country/Region",

                 size='Confirmed',log_y=True, log_x=True)

fig.show()
df1.head()
data=df1



data['Date']=pd.to_datetime(data.Date,dayfirst=True)

data.head()
data=data.rename(columns={'Date':'date',

                     'State/UnionTerritory':'state',

                         'Deaths':'deaths'})
latest = data[data['date'] == max(data['date'])].reset_index()

latest_grouped = latest.groupby('state')['Confirmed','deaths'].sum().reset_index()

latest = data[data['date'] == max(data['date'])]

latest = latest.groupby('state')['Confirmed', 'deaths'].max().reset_index()
latest.head()
latest.sort_values('Confirmed')
sns.barplot(x='deaths', y='state',  data=latest) 
sns.barplot(x='Confirmed', y='state',  data=latest) 
India_coord = pd.read_csv('/kaggle/input/coronavirus-cases-in-india/Indian Coordinates.csv')

data.head()
India_coord.head()