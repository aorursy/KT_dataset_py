# Loading the basic libraries



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Listing files in the input directory.

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Load plotly related packages

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go
# Read the required dataset

# I have read the 31st March version of the dataset. Kindly read the latest version 

testsC = pd.read_csv('/kaggle/input/covid19-tests-conducted-by-country/Tests_Conducted_31Mar2020.csv')

print(testsC.shape)

testsC.head()
# Get basic statistics for all Tests & confirmed cases (Positive)

testsC[['Tests', 'Positive', 'Tests/ million', 'Positive/ thousands' ]].describe()
# Create first scatter plot for all countries

trace1 = go.Scatter(

                    x = testsC['Tests/ million'],

                    y = testsC['Positive/ thousands'],

                    mode = "markers",

                    name = "Country or Region",

                    marker = dict(color = 'rgba(255, 50, 50, 0.8)'),

                    text= testsC['Country or region'])



data = [trace1]

layout = dict(title = 'Tests to Confirmed Cases per million of population',

              xaxis= dict(title= 'Tests/ million',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Confirmed Cases/ million',ticklen= 5,zeroline= False)

             )



fig = dict(data = data, layout = layout)

iplot(fig)
USA = testsC[testsC['Country or region'].str.contains('United States')]

print("Number of cases for United States: ", len(USA))

Italy = testsC[testsC['Country or region'].str.contains('Italy')]

print("Number of cases for United States: ", len(Italy))

China = testsC[testsC['Country or region'].str.contains('China')]

print("Number of cases for United States: ", len(China))
dropIndex = list(USA.index)

dropIndex.extend(list(Italy.index))

dropIndex.extend(list(China.index))

print(len(dropIndex))

World = testsC.drop(dropIndex,axis=0)

World.shape
usa = go.Scatter(

                    x = USA['Tests/ million'],

                    y = USA['Positive/ thousands'],

                    mode = "markers",

                    name = "United States",

                    marker = dict(color = 'rgba(255, 50, 50, 0.8)'),

                    text= USA['Country or region'])



italy = go.Scatter(

                    x = Italy['Tests/ million'],

                    y = Italy['Positive/ thousands'],

                    mode = "markers",

                    marker = dict(color = 'rgba(0, 50, 255, 0.8)'),

                    name = "Italy",

                    text= Italy['Country or region'])





china = go.Scatter(

                    x = China['Tests/ million'],

                    y = China['Positive/ thousands'],

                    mode = "markers",

                    marker = dict(color = 'rgba(0, 0, 0, 0.8)'),

                    name = "China",

                    text= China['Country or region'])





world = go.Scatter(

                    x = World['Tests/ million'],

                    y = World['Positive/ thousands'],

                    mode = "markers",

                    marker = dict(color = 'rgba(150, 120, 50, 0.8)'),

                    name = "World",

                    text= World['Country or region'])



data = [usa, italy, china, world]

layout = dict(title = 'Tests to Confirmed Cases per million of population',

              xaxis= dict(title= 'Tests/ million',ticklen= 25,zeroline= False),

              yaxis= dict(title= 'Confirmed Cases/ million',ticklen= 25,zeroline= False)

             )



fig = dict(data = data, layout = layout)

iplot(fig)
testsC['Tests/Positive'] = testsC['Tests'] / testsC['Positive']

testsC.sort_values(by=['Tests/Positive'], ascending=False, inplace=True)
import plotly.graph_objs as go



trace1 = go.Bar(

                x = testsC['Country or region'],

                y = testsC['Tests/Positive'],

                name = "Tests per Confirmed Cases(Positive)",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = testsC['Country or region'])



data = [trace1]



layout = go.Layout(barmode = "group")



fig = go.Figure(data = data, layout = layout)

iplot(fig)

testsC[testsC['Country or region'].str.contains('India')]