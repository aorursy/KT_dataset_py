from IPython.display import YouTubeVideo

YouTubeVideo('PLcE3AI9wwE', width=800, height=300)
# Data Manipulation Libraries

import numpy as np

import pandas as pd



# Plotting Libraries

import plotly.express as px

from iso3166 import countries

from statsmodels.tsa.arima_model import ARIMA

import matplotlib

import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from collections import OrderedDict

from statsmodels.tsa.seasonal import seasonal_decompose
df = pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv')



df.columns = ['Unnamed: 0', 'Unnamed: 0.1', 'Company Name', 'Location', 'Datum', 'Detail', 'Status Rocket', 'Rocket', 'Status Mission']

df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)



df.head()
df.info()
# Calculating

percent_missing = df.isnull().sum() * 100 / len(df)

missing_value_df = pd.DataFrame({'column': df.columns,

                                 'percent': percent_missing})

missing_value_df.sort_values('percent', inplace=True)

missing_value_df.reset_index(drop=True, inplace=True)

missing_value_df = missing_value_df[missing_value_df['percent']>0]



# Plotting

fig = px.bar(

    missing_value_df, 

    x='percent', 

    y="column", 

    orientation='h', 

    title='Columns with Missing Values', 

    height=200, 

    width=600

)

fig.show()
ds = df['Company Name'].value_counts().reset_index()

ds.columns = ['company', 'number of starts']

ds = ds.sort_values(['number of starts'],  ascending=False)

fig = px.bar(ds, x="company", y='number of starts', 

    

    color="number of starts",

    orientation='v', 

    title='Number of Launches by Company'

)

fig.update_layout(title_x=0.5, xaxis_title = 'Companies', yaxis_title = 'Number of Starts')

fig.update_xaxes(tickangle=60)

fig.show()


ds = df['Status Mission'].value_counts().reset_index()

ds.columns = ['mission_status', 'count']







fig = px.pie(

    ds, 

    values='count', 

    names="mission_status", 

    title='Success and Failure of the Rocket Missions', 

    width=500, 

    height=500,

    hole = 0.4

)



fig.update_layout( title_x = 0.5)

fig.show()







fig = px.bar(

    ds, 

    x='mission_status', 

    y="count",

    color = "count",

    orientation='v',  

    width=500

)

fig.update_layout(title = 'Mission Status Distribution', title_x=0.5, 

                  xaxis_title = 'Success or Failure', 

                  yaxis_title = 'Number of Starts')

fig.update_xaxes(tickangle=30)

fig.show()
ds = df['Status Rocket'].value_counts().reset_index()

ds.columns = ['status', 'count']

fig = px.pie(

    ds, 

    values='count', 

    names="status", 

    title='Rocket status', 

    width=500, 

    height=500,

    hole = 0.5

)

fig.show()
df['Rocket'] = df['Rocket'].fillna(0.0).str.replace(',', '')

df['Rocket'] = df['Rocket'].astype(np.float64).fillna(0.0)

df['Rocket'] = df['Rocket'] * 1000000



df.loc[df['Rocket']>4000000000, 'Rocket'] = 0.0

fig = px.histogram(

    df, 

    "Rocket", 

    nbins=50, 

    width=700, 

    height=500

)

fig.update_layout(    title='Rocket Value Distribution', title_x=0.5, 

                  xaxis_title = 'Value of Rockets', 

                  yaxis_title = 'Counts')

fig.show()

df['country'] = df['Location'].str.split(', ').str[-1]



countries_dict = {

    'Russia' : 'Russian Federation',

    'New Mexico' : 'USA',

    "Yellow Sea": 'China',

    "Shahrud Missile Test Site": "Iran",

    "Pacific Missile Range Facility": 'USA',

    "Barents Sea": 'Russian Federation',

    "Gran Canaria": 'USA'

}



df['country'] = df['country'].replace(countries_dict)
sun = df.groupby(['country', 'Company Name', 'Status Mission'])['Datum'].count().reset_index()

sun.columns = ['country', 'company', 'status', 'count']

fig = px.sunburst(

    sun, 

    path=[

        'country', 

        'company', 

        'status'

    ], 

    values='count', 

    title='Success of all countries at a glance',

    width=600,

    height=600

)

fig.update_layout(title_x = 0.5)

fig.show()
country_dict = {}

for c in countries:

    country_dict[c.name] = c.alpha3

    

df['alpha3'] = df['country']

df = df.replace({"alpha3": country_dict})



# Handling two exceptions. 

df.loc[df['country'] == "North Korea", 'alpha3'] = "PRK"

df.loc[df['country'] == "South Korea", 'alpha3'] = "KOR"







# Country Plotting Utility Function

def plot_map(dataframe, target_column, title, width=800, height=600):

    mapdf = dataframe.groupby(['country', 'alpha3'])[target_column].count().reset_index()

    fig = px.choropleth(

        mapdf, 

        locations="alpha3", 

        hover_name="country", 

        color=target_column, 

        projection="natural earth", 

        title=title

    )

    fig.update_layout(title = title, title_x = 0.5)

    fig.show()
plot_map(df, 'Status Mission', 'Number of starts per country')
fail_df = df[df['Status Mission'] == 'Failure']

plot_map(fail_df, 'Status Mission', 'Number of Fails per country')
# Data Preparation 

success_ratio = df.groupby(['country', 'Status Mission']).count()['Datum'].unstack().fillna(0).reset_index().copy()

total_attempts = success_ratio.iloc[:, 1:6].values.sum(axis = 1)

success_ratio['ratio'] = success_ratio['Success'] / total_attempts 

success_ratio.sort_values(['ratio'], ascending = False, inplace =True)



# Plotting

fig = px.bar(success_ratio, x="country", y='ratio',  color="ratio",

    orientation='v', 

    title='Ratio of Success for the Countries', 

    color_continuous_scale=px.colors.sequential.Viridis_r

)

fig.update_layout(title_x=0.5, xaxis_title = 'Countries', yaxis_title = 'Percentage of Success')

fig.update_xaxes(tickangle=60)

fig.show()
df[df['country'] == 'Kenya']
data = df.groupby(['country'])['Rocket'].sum().reset_index()

data = data[data['Rocket'] > 0]



data.columns = ['country', 'money']

data.sort_values(['money'], ascending = False, inplace = True)

fig = px.bar(

    data, 

    x='country', 

    y="money",

    color = "money", 

    orientation='v',  

    color_continuous_scale=px.colors.sequential.Viridis_r

)

fig.update_layout(title='Country-wise Expenditure',title_x=0.5, xaxis_title = 'Countries', yaxis_title = 'Total Money Spent')

fig.update_xaxes(tickangle=30)

fig.show()
# Calculation

money = df.groupby(['country'])['Rocket'].sum()

starts = df['country'].value_counts().reset_index()

starts.columns = ['country', 'count']



av_money_df = pd.merge(money, starts, on='country')

av_money_df['avg'] = av_money_df['Rocket'] / av_money_df['count']

av_money_df = av_money_df[av_money_df['avg']>0]

av_money_df = av_money_df.reset_index()

av_money_df.sort_values(['avg'], ascending = False, inplace = True)



# Plotting

fig = px.bar(

    av_money_df, 

    x='country', 

    y="avg",

    color = 'avg',

    orientation='v', 

    color_continuous_scale=px.colors.sequential.Viridis_r

)

fig.update_layout(title='Average money per one launch', title_x=0.5, xaxis_title = 'Countries', yaxis_title = 'Money Spent per Mission')

fig.update_xaxes(tickangle=30)

fig.show()
data = df.groupby(['Company Name'])['Rocket'].sum().reset_index()

data = data[data['Rocket'] > 0]

data.columns = ['company', 'money']

data.sort_values(['money'], ascending = False, inplace = True)

fig = px.bar(

    data, 

    x='company', 

    y="money",

    color = "money",

    orientation='v',

    color_continuous_scale=px.colors.sequential.Viridis_r)

fig.update_layout(    title='Total money spent on missions', title_x=0.5, xaxis_title = 'Companies', yaxis_title = 'Money Spent in Total')

fig.update_xaxes(tickangle=60)

fig.show()
money = df.groupby(['Company Name'])['Rocket'].sum()

starts = df['Company Name'].value_counts().reset_index()

starts.columns = ['Company Name', 'count']



av_money_df = pd.merge(money, starts, on='Company Name')

av_money_df['avg'] = av_money_df['Rocket'] / av_money_df['count']

av_money_df = av_money_df[av_money_df['avg']>0]

av_money_df = av_money_df.reset_index()

av_money_df.sort_values(['avg'], ascending = False, inplace = True)



fig = px.bar(

    av_money_df, 

    x='Company Name', 

    y="avg", 

    color = 'avg',

    orientation='v',

    color_continuous_scale=px.colors.sequential.Viridis_r

)

fig.update_layout(       title='Average money per one launch', title_x=0.5, xaxis_title = 'Companies', yaxis_title = 'Money Spent per Launch')

fig.update_xaxes(tickangle=60)

fig.show()
df['date'] = pd.to_datetime(df['Datum'])

df['year'] = df['date'].apply(lambda datetime: datetime.year)

df['month'] = df['date'].apply(lambda datetime: datetime.month)

df['weekday'] = df['date'].apply(lambda datetime: datetime.weekday())
ds = df.groupby(['Company Name'])['year'].nunique().reset_index()

ds.columns = ['company', 'count']

ds.sort_values(['count'], inplace = True, ascending = False)

fig = px.bar(

    ds, 

    x="company", 

    y="count",

    color = "count", 

    color_continuous_scale=px.colors.sequential.Viridis_r

)



fig.update_layout( title='Most Experienced Companies (years of experience)', title_x=0.5, xaxis_title = 'Companies', yaxis_title = 'Years of Experience')

fig.update_xaxes(tickangle=60)

fig.show()
res = []

for group in df.groupby(['Company Name']):

    res.append(group[1][['Company Name', 'year']].head(1))

data = pd.concat(res)

data = data.sort_values('year', ascending = False)

data['year'] = 2020 - data['year']

fig = px.bar(

    data, 

    x="year", 

    y="Company Name",

    color = 'year',

    orientation='h', 

    color_continuous_scale=px.colors.sequential.Viridis_r, height = 1000

)

fig.update_layout( title='Years from last mission', title_x=0.5, xaxis_title = 'Years after last mission', yaxis_title = 'Companies Name')



fig.show()
data