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
#Importing Necessary Libraries

import matplotlib.pyplot as plt

import plotly.graph_objects as go 

import seaborn as sns

import plotly

import plotly.express as px

from fbprophet.plot import plot_plotly

from fbprophet import Prophet
#Setting up plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot_mpl

import plotly.offline as py

init_notebook_mode(connected=True)
# Reading dataset

dataset = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
dataset
# Checking if there is any null values

dataset.isnull().any()
# Seeing the datset with null rows

dataset[dataset.isnull().any(axis=1)]
# Affected Countries

print(f"Affected Countries are : {dataset['Country/Region'].unique()}")

print(f"Total Affected Countries are : {len(dataset['Country/Region'].unique())}")
# Affected States/Provinces

print(f"Affected State/Provinces are : {dataset['Province/State'].unique()}")

print(f"Total Affected State/Provinces are : {len(dataset['Province/State'].unique())}")
fig = px.bar(dataset, x='ObservationDate', y='Confirmed', hover_data=['Province/State', 'Deaths', 'Recovered'], color='Country/Region')

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Confirmed bar plot for each country',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
fig = px.bar(dataset.loc[dataset['Country/Region'] == 'Mainland China'], x='ObservationDate', y='Confirmed', hover_data=['Province/State', 'Deaths', 'Recovered'], color='Province/State')

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Confirmed bar plot for Mainland China',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
# Saving the image 

py.plot(fig, filename='Confirmed_plot.html')
fig = px.bar(dataset, x='ObservationDate', y='Deaths', hover_data=['Province/State', 'Confirmed', 'Recovered'], color='Country/Region')

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Death bar plot for each country',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
fig = px.bar(dataset.loc[dataset['Country/Region'] == 'Mainland China'], x='ObservationDate', y='Deaths', hover_data=['Province/State', 'Confirmed', 'Recovered'], color='Province/State')

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Death bar plot for Mainland China',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
fig = px.line(dataset, x="ObservationDate", y="Confirmed", color='Country/Region', hover_data=['Province/State', 'Deaths'])

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Confirmed Plot for each Country',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
fig = px.line(dataset, x="ObservationDate", y="Deaths", color='Country/Region', hover_data=['Province/State', 'Deaths'])

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Death plot for each country',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
fig = px.line(pd.DataFrame(dataset.groupby('ObservationDate')['Confirmed'].sum().reset_index()), x="ObservationDate", y="Confirmed")

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Showing Deaths of total country',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
pxdf = px.data.gapminder()

country_isoAlpha = pxdf[['country', 'iso_alpha']].drop_duplicates()

country_isoAlpha.rename(columns = {'country':'Country'}, inplace=True)

country_isoAlpha.set_index('Country', inplace=True)

country_map = country_isoAlpha.to_dict('index')
def getCountryIsoAlpha(country):

    try:

        return country_map[country]['iso_alpha']

    except:

        return country
dataset.replace({'Country/Region': 'Mainland China'}, 'China', inplace=True)

dataset['iso_alpha'] = dataset['Country/Region'].apply(getCountryIsoAlpha)
df_plot = dataset.groupby('iso_alpha').max().reset_index()

fig = px.choropleth(df_plot, locations="iso_alpha",

                    color="Confirmed",

                    hover_data=["Confirmed", "Deaths", "Recovered"],

                    color_continuous_scale="Viridis")

fig.update_geos(fitbounds="locations", visible=True)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, title_text = 'Deaths Cases in World')



fig.show()
fig = px.scatter_geo(dataset, locations="iso_alpha", 

                     color="Confirmed", size='Confirmed', hover_name="Country/Region", 

                    hover_data=["Confirmed", "Deaths", "Recovered"],

                     projection="natural earth", animation_frame="ObservationDate")

fig.show()


fig = px.choropleth(df_plot, locations="iso_alpha",

                    color="Deaths",

                    hover_data=["Confirmed", "Deaths", "Recovered"],

                    color_continuous_scale="Viridis")

fig.update_geos(fitbounds="locations", visible=True)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, title_text = 'Deaths Cases in World')



fig.show()
fig = px.scatter_geo(dataset, locations="iso_alpha", 

                     color="Deaths", size='Deaths', hover_name="Country/Region",

                    hover_data=["Confirmed", "Deaths", "Recovered"] ,

                     projection="natural earth", animation_frame="ObservationDate")

fig.show()
confirmed_training_dataset = pd.DataFrame(dataset.groupby('ObservationDate')['Confirmed'].sum().reset_index()).rename(columns={'ObservationDate': 'ds', 'Confirmed': 'y'})

confirmed_training_dataset
# Making the Model

prophet = Prophet()

prophet.fit(confirmed_training_dataset)

future = prophet.make_future_dataframe(periods=20)

confirmed_forecast = prophet.predict(future)
fig = plot_plotly(prophet, confirmed_forecast)  

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Predictions for Confirmed',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig
death_training_dataset = pd.DataFrame(dataset.groupby('ObservationDate')['Deaths'].sum().reset_index()).rename(columns={'ObservationDate': 'ds', 'Deaths': 'y'})

death_training_dataset
# Making the Model

prophet = Prophet()

prophet.fit(death_training_dataset)

future = prophet.make_future_dataframe(periods=20)

deaths_forecast = prophet.predict(future)
fig = plot_plotly(prophet, deaths_forecast)  

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Predictions of Deaths',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig
# Saving Death Forcasting  

py.plot(fig, filename='death_forcasting.html')
ontario_data = dataset.loc[dataset['Province/State'] == 'Ontario']

fig = px.bar(ontario_data, x='ObservationDate', y='Confirmed', hover_data=['Deaths', 'Recovered'])

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Confirmed bar plot for Ontario',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
ontario_data = dataset.loc[dataset['Province/State'] == 'Ontario']

fig = px.bar(ontario_data, x='ObservationDate', y='Deaths', hover_data=['Confirmed', 'Recovered'])

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Deaths bar plot for Ontario',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
ontario_training_dataset = pd.DataFrame(ontario_data.groupby('ObservationDate')['Deaths'].sum().reset_index()).rename(columns={'ObservationDate': 'ds', 'Deaths': 'y'})
ontario_training_dataset
# Making the Model

prophet = Prophet()

prophet.fit(ontario_training_dataset.iloc[9:, :])

future = prophet.make_future_dataframe(periods=2)

ontario_deaths_forecast = prophet.predict(future)
fig = plot_plotly(prophet, ontario_deaths_forecast)  

fig