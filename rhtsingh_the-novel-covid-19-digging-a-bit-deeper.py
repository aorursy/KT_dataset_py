from __future__ import absolute_import, print_function, unicode_literals, division

from builtins import range, input
!ls -GFlash --color ../input
%%time

%load_ext autoreload

%autoreload 2

%matplotlib inline

%config InlineBackend.figure_format = "retina"



import os

import gc

gc.enable()

import sys

import csv

import glob

import json



import numpy as np

import pandas as pd

from scipy import stats

import seaborn as sns 

import matplotlib.pyplot as plt



import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots

import folium 

from folium import plugins



plt.rc('font', size=12) 

plt.rc('axes', titlesize=14)

plt.rc('axes', labelsize=12) 

plt.rc('xtick', labelsize=12)

plt.rc('ytick', labelsize=12) 

plt.rc('legend', fontsize=12) 

plt.rc('figure', titlesize=14) 

plt.rcParams['figure.dpi'] = 60

sns.set(rc={"figure.figsize":(12,10)})

sns.set_style("whitegrid")

sns.set_context("notebook")

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]

sns.set_palette(sns.xkcd_palette(colors))



import datetime

import multiprocessing

import missingno as msno 

from pathlib import Path

from tqdm import tqdm_notebook

from IPython.display import display, clear_output

import warnings

warnings.simplefilter('ignore')
# function to display summary statistics

def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary
print("Display input files")

data_dir = Path('../input/covid19-global-forecasting-week-2/')

os.listdir(data_dir)
print("Read data and display")

data = pd.read_csv(data_dir/'train.csv', parse_dates=['Date'])

data.rename(columns={'Date': 'date', 

                     'Id': 'id',

                     'Province/State':'state',

                     'Country/Region':'country',

                     'Lat':'lat',

                     'Long': 'long',

                     'ConfirmedCases': 'confirmed',

                     'Fatalities':'deaths',

                    }, inplace=True)

with pd.option_context("display.max_rows",10,"display.max_columns",100):

    display(data)
print("Display External Data")

cleaned_data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])

cleaned_data.rename(columns={'Date': 'date', 

                     'Id': 'id',

                     'Province/State':'state',

                     'Country/Region':'country',

                     'Lat':'lat',

                     'Long': 'long',

                     'ConfirmedCases': 'confirmed',

                     'Fatalities':'deaths',

                    }, inplace=True)

with pd.option_context("display.max_rows",10,"display.max_columns",100):

    display(cleaned_data)
print("External Data Summary")

resumetable(cleaned_data)
# check for missing values

msno.bar(cleaned_data, sort='descending')
# create a new column of active cases 

cases = ['Confirmed','Deaths','Recovered','Active']

cleaned_data['Active'] = cleaned_data['Confirmed'] - cleaned_data['Deaths'] - cleaned_data['Recovered']



# replace 'mainland china' with 'china'

cleaned_data['country'] = cleaned_data['country'].replace('Mainland China', 'China')



# fill blank in state column for null values

cleaned_data[['state']] = cleaned_data[['state']].fillna('')



# fill 0 in cases for null values

cleaned_data[cases] = cleaned_data[cases].fillna(0)



# rename Date to date

cleaned_data.rename(columns={'Date':'date'}, inplace=True)

df = cleaned_data
print("External Data")

print(f"Earliest Entry: {df['date'].min()}")

print(f"Last Entry:     {df['date'].max()}")

print(f"Total Days:     {df['date'].max() - df['date'].min()}")
formated_gdf = df.groupby(['date', 'country'])['Confirmed', 'Deaths'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 

                     color="Confirmed", size='size', hover_name="country", 

                     range_color= [0, 1500], 

                     projection="natural earth", animation_frame="date", 

                     title='COVID-19: Spread Over Time', color_continuous_scale="portland")

fig.show()
# %%time

# world_map = folium.Map(location=[0, 0], zoom_start=2.5, tiles='Stamen Terrain')

# tooltip = 'Click me!'

# for lat, lon, Confirmed, Deaths, Recovered, name in zip(df['lat'], df['long'], df['Confirmed'],df['Deaths'],df['Recovered'], df['country']):

#     folium.Marker([lat, lon],

                        

#                         popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

#                                 '<strong>Confirmed Cases</strong>: ' + str(Confirmed) + '<br>'

#                                 '<strong>Recovered Cases</strong>: ' + str(Recovered) +'<br>'

#                                 '<strong>Deaths Cases</strong>: ' + str(Deaths) +'<br>'),

#                         icon=folium.Icon(color='red', icon='info-sign'),

#                         tooltip=tooltip

#                         ).add_to(world_map)

# world_map
plt.figure(figsize=(16,28))

summary = df.groupby("country")["Confirmed","Deaths","Active","Recovered"].sum().reset_index().sort_values(by="Confirmed",ascending=False).reset_index(drop=True)[:10][::-1]



fig = go.Figure()

fig.add_trace(go.Bar(name='Confirmed Cases', y=list(summary["country"]), x=list(summary["Confirmed"]),orientation='h'))

fig.update_layout(title="Top 10 Affected Countries")

fig.show()



fig = go.Figure()

fig.add_trace(go.Bar(name='Active Cases', y=list(summary["country"]), x=list(summary["Active"]),orientation='h')),

fig.add_trace(go.Bar(name='Recovered Cases', y=list(summary["country"]), x=list(summary["Recovered"]),orientation='h'))

fig.add_trace(go.Bar(name='Death Cases', y=list(summary["country"]), x=list(summary["Deaths"]),orientation='h')),

fig.update_layout(barmode='stack',title="Active, Recovered & Deaths in Top 10 Countries")

fig.show()
grouped = df.groupby('date')['Recovered', 'Deaths', 'Active'].sum().reset_index()



temp = grouped.melt(id_vars="date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='case', value_name='count')

fig = px.area(temp, x="date", y="count", color='case',

             title='Cases - Total Cases: Area Plot', color_discrete_sequence = ['cyan', 'red', 'orange'], labels={'count':'Comparisons', 'date':''})

fig.show()
grouped = df.groupby('date')['date', 'Confirmed'].sum().reset_index()



fig = px.line(grouped, x="date", y="Confirmed", 

              title="Worldwide Confirmed Cases Over Time", labels={'date':'', 'Confirmed':'Total Coronavirus Cases'})

fig.show()



fig = px.line(grouped, x="date", y="Confirmed", 

              title="Worldwide Confirmed Cases (Logarithmic Scale) Over Time", 

              log_y=True,labels={'date':'', 'Confirmed':'Total Coronavirus Cases'})

fig.show()
grouped = df.groupby(["date"])["Confirmed"].sum().reset_index()

grouped['daily_cases'] = grouped['Confirmed'].shift(-1) - grouped['Confirmed']

grouped['daily_cases'].fillna(0)

fig = px.bar(grouped, x='date',y='daily_cases', labels={'date':'', 'daily_cases':'Novel Coronavirus Daily Cases'},title='Daily New Cases')

fig.show()
grouped["growth_factor"] = grouped["daily_cases"].shift(-1) / grouped["daily_cases"]

grouped["growth_factor"]=grouped["growth_factor"].fillna(0)

fig = go.Figure()

fig.add_trace(go.Scatter(x=grouped['date'], y=grouped["growth_factor"],

                    mode='lines+markers',

                    name='lines+markers'))

fig.update_layout(

    title={

        'text': "Growth Factor",

})

fig.show()
grouped = df[df["country"]!="China"].groupby('date')['date', 'Confirmed'].sum().reset_index()



fig = px.line(grouped, x="date", y="Confirmed", 

              title="Total Cases outside of China (Linear Scale)", labels={'date':'', 'Confirmed':'Total Coronavirus Cases'})

fig.show()



fig = px.line(grouped, x="date", y="Confirmed", 

              title="Total Cases outside of China (Logarithmic Scale)", 

              log_y=True, labels={'date':'', 'Confirmed':'Total Coronavirus Cases'})

fig.show()
grouped = df[df["country"]!="China"].groupby(["date"])["Confirmed"].sum().reset_index()

grouped['daily_cases'] = grouped['Confirmed'].shift(-1) - grouped['Confirmed']

grouped['daily_cases'].fillna(0)

fig = px.bar(grouped, x='date',y='daily_cases', labels={'date':'', 'daily_cases':'Novel Coronavirus Daily Cured'},title='Daily Cured')

fig.show()
grouped["growth_factor"] = grouped["daily_cases"].shift(-1) / grouped["daily_cases"]

grouped["growth_factor"]=grouped["growth_factor"].fillna(0)

fig = go.Figure()

fig.add_trace(go.Scatter(x=grouped['date'], y=grouped["growth_factor"],

                    mode='lines+markers',

                    name='lines+markers'))

fig.update_layout(

    title={

        'text': "Growth Factor",

})

fig.show()
plt.figure(figsize=(8,8))

grouped = df.groupby("country")["Confirmed"].sum().reset_index()



temp = pd.DataFrame()

temp["China"] = grouped[grouped["country"]=="China"]["Confirmed"]

temp["Others"] = grouped[grouped["country"]!="China"]["Confirmed"].sum()

temp = temp.melt()



fig = px.pie(temp, values='value', names='variable', title='Distribution of Cases WorldWide',color='variable', color_discrete_map={'Others':'lightcyan',

                                 'China':'royalblue'}, hover_data=['value'], labels={'value':'Total Affected','variable':'Country'})

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
formated_gdf = df.groupby(['date', 'country'])['Confirmed', 'Deaths', 'Active', 'Recovered'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['Active'].pow(0.3)

formated_gdf['size'].fillna(formated_gdf['size'].mean(),inplace=True)



fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 

                     color="Active", size='size', hover_name="country", 

                     range_color= [0, 1000], 

                     projection="natural earth", animation_frame="date", 

                     title='COVID-19: Active Cases Over Time', color_continuous_scale="portland")

fig.update(layout_coloraxis_showscale=False)

fig.show()
grouped_df = df.groupby('date')['Active'].sum().reset_index()

fig = go.Figure()

fig.add_trace(go.Scatter(x=grouped_df['date'], y=grouped_df['Active'],

                    mode='lines+markers',

                    name='Active Cases'))

fig.update_layout(

    title={

        'text': "Active Cases",

})
grouped_df = df.groupby('date')['Recovered'].sum().reset_index()

fig = go.Figure()

fig.add_trace(go.Scatter(x=grouped_df['date'], y=grouped_df['Recovered'],

                    mode='lines+markers',

                    name='Recovered Cases'))

fig.update_layout(

    title={

        'text': "Total Cured",

})
grouped = df.groupby(["date"])["Recovered"].sum().reset_index()

grouped['daily_cases'] = grouped['Recovered'].shift(-1) - grouped['Recovered']

grouped['daily_cases'].fillna(0)

fig = px.bar(grouped, x='date',y='daily_cases',labels={'date':'', 'daily_cases':'Novel Coronavirus Daily Cured'},title='Daily Cured')



fig.show()
grouped_df = df.groupby('date')['Recovered','Deaths'].sum().reset_index()

fig = go.Figure()

fig.add_trace(go.Scatter(x=grouped_df['date'], y=grouped_df['Recovered'],

                    mode='lines+markers',

                    name='Recovered Cases'))

fig.add_trace(go.Scatter(x=grouped_df['date'], y=grouped_df['Deaths'],

                    mode='lines+markers',

                    name='Death Cases'))



fig.update_layout(

    title={

        'text': "Outcome of Cases (Recovery or Death)",

})
grouped = df.groupby(["date"])["Recovered","Deaths"].sum().reset_index()

grouped['daily_cases_rec'] = grouped['Recovered'].shift(-1) - grouped['Recovered']

grouped['daily_cases_death'] = grouped['Deaths'].shift(-1) - grouped['Deaths']



fig = go.Figure()

fig.add_trace(go.Scatter(x=grouped['date'], y=grouped['daily_cases_rec'],

                    mode='lines+markers',

                    name='Recovered Cases'))

fig.add_trace(go.Scatter(x=grouped['date'], y=grouped['daily_cases_death'],

                    mode='lines+markers',

                    name='Death Cases'))



fig.update_layout(

    title={

        'text': "Outcome of Cases everyday (Recovery or Death)",

})

fig.show()
formated_gdf = df.groupby(['date', 'country'])['Confirmed', 'Deaths'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['Deaths'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 

                     color="Deaths", size='size', hover_name="country", 

                     range_color= [0, 100], 

                     projection="natural earth", animation_frame="date", 

                     title='COVID-19: Deaths Over Time', color_continuous_scale="peach")

# fig.update(layout_coloraxis_showscale=False)

fig.show()