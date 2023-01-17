# importing all necessary libraries

import pandas as pd

import numpy as np

from datetime import date

import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt

%matplotlib inline

import pycountry

import plotly.graph_objects as go
# Reading the dataset

coronaVirus_df =  pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv",index_col='ObservationDate', parse_dates=['ObservationDate'])

coronaVirus_df.tail()
coronaVirus_df.shape
coronaVirus_df.isnull().values.any()
coronaVirus_df.isnull().sum()
#replacing null values in Province/State with Country names

coronaVirus_df['Province/State'].fillna(coronaVirus_df['Country/Region'], inplace=True)
coronaVirus_df.drop(['SNo'], axis=1, inplace=True)
coronaVirus_df.head()
#creating new columns for date, month and time which would be helpful for furthur computation

coronaVirus_df['year'] = pd.DatetimeIndex(coronaVirus_df['Last Update']).year

coronaVirus_df['month'] = pd.DatetimeIndex(coronaVirus_df['Last Update']).month

coronaVirus_df['date'] = pd.DatetimeIndex(coronaVirus_df['Last Update']).day

coronaVirus_df['time'] = pd.DatetimeIndex(coronaVirus_df['Last Update']).time
coronaVirus_df.head()
coronaVirus_df.rename(columns={"Country/Region": "Country", "Province/State": "State"} , inplace=True)
# A look at the different cases - confirmed, death and recovered

print('Globally Confirmed Cases: ',coronaVirus_df['Confirmed'].sum())

print('Global Deaths: ',coronaVirus_df['Deaths'].sum())

print('Globally Recovered Cases: ',coronaVirus_df['Recovered'].sum())
coronaVirus_df[['Confirmed', 'Deaths', 'Recovered']].sum().plot(kind='bar', color = '#007bcc')
Recovered_percent = (coronaVirus_df['Recovered'].sum() / coronaVirus_df['Confirmed'].sum()) * 100

print("% of people recovered from virus: ",Recovered_percent)



Death_percent = (coronaVirus_df['Deaths'].sum()/coronaVirus_df['Confirmed'].sum()) * 100

print("% of people died due to virus:", Death_percent)

import plotly.graph_objects as go

grouped_multiple = coronaVirus_df.groupby(['ObservationDate']).agg({'Confirmed': ['sum']})

grouped_multiple.columns = ['Confirmed ALL']

grouped_multiple = grouped_multiple.reset_index()

fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=grouped_multiple['ObservationDate'], 

                         y=grouped_multiple['Confirmed ALL'],

                         mode='lines+markers',

                         name='Confirmed',

                         line=dict(color='red', width=2)))

fig.show()
# Total Number Of countries which are affected by the virus



countries= coronaVirus_df['Country'].unique()

total_countries= len(countries)

print('Total countries affected:',total_countries)

print('Countries affected are:',countries)
# Number of confirmed cases reported Country wise



global_confirmed_cases = coronaVirus_df.groupby('Country').sum().Confirmed

global_confirmed_cases.sort_values(ascending=False)
global_death_cases = coronaVirus_df.groupby('Country').sum().Deaths

global_death_cases.sort_values(ascending=False)
global_recovered_cases = coronaVirus_df.groupby('Country').sum().Recovered

global_recovered_cases.sort_values(ascending=False)
#plotting graphs for total Confirmed, Death and Recovery cases

plt.rcParams["figure.figsize"] = (12,9)

ax1 = coronaVirus_df[['month','Confirmed']].groupby(['month']).sum().plot()

ax1.set_ylabel("Total Number of Confirmed Cases")

ax1.set_xlabel("month")



#ax2 = coronaVirus_df[['date','Deaths', 'Recovered']].groupby(['date']).sum().plot()

#ax2.set_ylabel("Recovered and Deaths Cases")

#ax2.set_xlabel("date")
# Let's look the various Provinces/States affected



data_countryprovince = coronaVirus_df.groupby(['Country','State']).sum()

data_countryprovince.sort_values(by='Confirmed',ascending=False)
# Top Affected countries



top_affected_countries = global_confirmed_cases.sort_values(ascending=False)

top_affected_countries.head(5)
# Finding countries which are relatively safe due to less number of reported cases

top_unaffected_countries = global_confirmed_cases.sort_values(ascending=True)

top_unaffected_countries.head(5)
#Mainland China

China_data = coronaVirus_df[coronaVirus_df['Country']=='Mainland China']

China_data
x = China_data.groupby('State')['Confirmed'].sum().sort_values().tail(15)
x.plot(kind='barh', color='#86bf91')

plt.xlabel("Confirmed case Count", labelpad=14)

plt.ylabel("State", labelpad=14)

plt.title("Confirmed cases count in China states", y=1.02);
US_data = coronaVirus_df[coronaVirus_df['Country']=='US']

US_data
x = US_data.groupby('State')['Confirmed'].sum().sort_values(ascending=False).tail(20)

x
x.plot(kind='barh', color='#86bf91')

plt.xlabel("Confirmed case Count", labelpad=14)

plt.ylabel("States", labelpad=14)

plt.title("Confirmed cases count in US states", y=1.02);
India_data = coronaVirus_df[coronaVirus_df['Country']=='India']

India_data
# Using plotly.express

import plotly.express as px



import pandas as pd



fig = px.line(coronaVirus_df, x='Last Update', y='Confirmed')

fig.show()


fig = px.line(coronaVirus_df, x='Last Update', y='Deaths')

fig.show()
import pandas as pd

import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Scatter(

                x=coronaVirus_df['date'],

                y=coronaVirus_df['Confirmed'],

                name="Confirmed",

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=coronaVirus_df['date'],

                y=coronaVirus_df['Recovered'],

                name="Recovered",

                line_color='dimgray',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=coronaVirus_df['date'],

                y=coronaVirus_df['Deaths'],

                name="Deaths",

                line_color='red',

                opacity=0.8))



# Use date string to set xaxis range

fig.update_layout(xaxis_range=['2020-01-22','2020-03-10'],

                  title_text="Cases over time")

fig.show()
import pandas as pd

import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Scatter(

                x=coronaVirus_df['date'],

                y=coronaVirus_df['Recovered'],

                name="Recovered",

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=coronaVirus_df['date'],

                y=coronaVirus_df['Deaths'],

                name="Deaths",

                line_color='red',

                opacity=0.8))



# Use date string to set xaxis range

fig.update_layout(xaxis_range=['2020-01-22 00:00:00','2020-03-10 23:59:59'],

                  title_text="Recovered vs Deaths over time in China")

fig.show()
import pandas as pd

import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Scatter(

                x=coronaVirus_df.time,

                y=coronaVirus_df['Confirmed'],

                name="Confirmed",

                line_color='deepskyblue',

                opacity=0.8))



# Use date string to set xaxis range

fig.update_layout(xaxis_range=['2020-01-31','2020-02-03'],

                  title_text="Confirmed Cases over time")

fig.show()