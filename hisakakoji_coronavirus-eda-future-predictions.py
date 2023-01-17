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

dataset = pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
dataset.head(10)
# Checking if there is any null values

dataset.isnull().any()
# Seeing the datset with null rows

dataset[dataset.isnull().any(axis=1)]
# Affected Countries

print(f"Affected Countries are : {dataset['Country'].unique()}")

print(f"Total Affected Countries are : {len(dataset['Country'].unique())}")
# Affected States/Provinces

print(f"Affected State/Provinces are : {dataset['Province/State'].unique()}")

print(f"Total Affected State/Provinces are : {len(dataset['Province/State'].unique())}")
fig = px.bar(dataset, x='Date', y='Confirmed', hover_data=['Province/State', 'Deaths', 'Recovered'], color='Country')

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
fig = px.bar(dataset.loc[dataset['Country'] == 'Mainland China'], x='Date', y='Confirmed', hover_data=['Province/State', 'Deaths', 'Recovered'], color='Province/State')

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
fig = px.bar(dataset, x='Date', y='Deaths', hover_data=['Province/State', 'Confirmed', 'Recovered'], color='Country')

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
fig = px.bar(dataset.loc[dataset['Country'] == 'Mainland China'], x='Date', y='Deaths', hover_data=['Province/State', 'Confirmed', 'Recovered'], color='Province/State')

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
fig = px.line(dataset, x="Date", y="Confirmed", color='Country', hover_data=['Province/State', 'Deaths'])

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
fig = px.line(dataset, x="Date", y="Deaths", color='Country', hover_data=['Province/State', 'Deaths'])

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
fig = px.line(pd.DataFrame(dataset.groupby('Date')['Confirmed'].sum().reset_index()), x="Date", y="Confirmed")

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
datasetJP = dataset[ dataset['Country'] == 'Japan']
datasetJP
confirmed_training_dataset = pd.DataFrame(dataset.groupby('Date')['Confirmed'].sum().reset_index()).rename(columns={'Date': 'ds', 'Confirmed': 'y'})

confirmed_training_dataset
confirmed_training_datasetJP = pd.DataFrame(datasetJP.groupby('Date')['Confirmed'].sum().reset_index()).rename(columns={'Date': 'ds', 'Confirmed': 'y'})

confirmed_training_datasetJP
# Making the Model

prophet = Prophet()

#confirmed_training_dataset['cap'] = 1000000

prophet.fit(confirmed_training_dataset)

future = prophet.make_future_dataframe(periods=100)

#future['cap'] = 1000000

confirmed_forecast = prophet.predict(future)
confirmed_forecast
# Making the Model

prophetJP = Prophet()

prophetJP.fit(confirmed_training_datasetJP)

futureJP = prophetJP.make_future_dataframe(periods=100)

confirmed_forecastJP = prophetJP.predict(futureJP)
future
fig = plot_plotly(prophet, confirmed_forecast)  

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='世界　感染者数予測',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig
figJP = plot_plotly(prophetJP, confirmed_forecastJP)  

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='日本　感染者数予測',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

figJP.update_layout(annotations=annotations)

figJP
death_training_dataset = pd.DataFrame(dataset.groupby('Date')['Deaths'].sum().reset_index()).rename(columns={'Date': 'ds', 'Deaths': 'y'})

death_training_dataset
death_training_datasetJP = pd.DataFrame(datasetJP.groupby('Date')['Deaths'].sum().reset_index()).rename(columns={'Date': 'ds', 'Deaths': 'y'})
# Making the Model

prophet = Prophet()

prophet.fit(death_training_dataset)

future = prophet.make_future_dataframe(periods=30)

deaths_forecast = prophet.predict(future)
# Making the Model

prophetJP = Prophet()

prophetJP.fit(death_training_datasetJP)

futureJP = prophetJP.make_future_dataframe(periods=30)

deaths_forecastJP = prophetJP.predict(futureJP)
fig = plot_plotly(prophet, deaths_forecast)  

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='世界での死亡者数',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig
figJP = plot_plotly(prophetJP, deaths_forecastJP)  

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Predictions of Deaths JP',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

figJP.update_layout(annotations=annotations)

figJP
# Saving Death Forcasting  

py.plot(fig, filename='death_forcasting.html')
data = {'Day':['Feb 5', 'Feb 6', 'Feb 7', 'Feb 8', 'Feb 9', 'Feb 10'],

    'Predicted Values':  [529, 572, 614, 657, 700, 743],

        'Actual Values': [494, 634, 638, 813, 910, 1013],

        }



predictions_accuracy = pd.DataFrame(data)
fig = go.Figure()

fig.add_trace(go.Scatter(x=['Feb 5', 'Feb 6', 'Feb 7', 'Feb 8', 'Feb 9', 'Feb 10'], y=[529, 572, 614, 657, 700, 743],

                    mode='lines',

                    name='Predicted Values'))

fig.add_trace(go.Scatter(x=['Feb 5', 'Feb 6', 'Feb 7', 'Feb 8', 'Feb 9', 'Feb 10'], y=[494, 634, 638, 813, 910, 1013],

                    mode='lines+markers',

                    name='Actual Values'))

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Predictions and Actual Death Data',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()