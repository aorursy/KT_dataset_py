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

testsC = pd.read_csv('/kaggle/input/covid19-tests-conducted-by-country/Tests_conducted_23May2020.csv')

print(testsC.shape)

#testsC.head()
#list(testsC.columns.values)
#Define additional columns   

# Pre-existing columns Tests" = absolute tests, "Positive"= absol positive tests, "Tests/ million"= freq of testing

# New calculated columns:  "pop"= region population in millions, "dp"= disease prevalence = Positive/pop = positive tests/million population, 

# "test_Ratio"= Tests/Positive

testsC['pop'] = testsC['Tested']/testsC['Tested\u2009/millionpeople']

testsC['dp'] = testsC['Positive']/testsC['pop']

testsC['test_Ratio'] =testsC['Tested']/testsC['Positive']

#testsC.head()
import plotly.graph_objs as go

testsC.sort_values(by=['Tested\u2009/millionpeople'], ascending=False, inplace=True)

trace1 = go.Bar(

                x = testsC['Country'],

                y = testsC['Tested\u2009/millionpeople'],

                name = "Disease Prevalence/Testing Ratio",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = testsC['Country'])



data = [trace1]

             

layout = go.Layout(barmode = "group")



fig = go.Figure(data = data, layout = layout)

iplot(fig)

# Create  scatter plot for all countriesshowing  tests done per million of population versus 

# prevalence of disease as positive tests per million population



trace1 = go.Scatter(

                    y = testsC['Tested\u2009/millionpeople'],

                    x = testsC['dp'],

                    mode = "markers",

                    name = "Country",

                    marker = dict(color = 'rgba(255, 50, 50, 0.8)'),

                    text= testsC['Country'])



data = [trace1]

layout = dict(title = 'Tests Done per Million Populaton versus Disease Prevalence',

              xaxis= dict(title= 'Disease Prevalence (Positive Tests/Million Population' ,ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Tests Done per million population',ticklen= 5,zeroline= False)

             )



fig = dict(data = data, layout = layout)

iplot(fig)

                    
# Create  scatter plot for all countriesshowing  testing ratio versus 

# prevalence of disease as positive tests per million population



trace1 = go.Scatter(

                    y = testsC['test_Ratio'],

                    x = testsC['dp'],

                    mode = "markers",

                    name = "Country",

                    marker = dict(color = 'rgba(255, 50, 50, 0.8)'),

                    text= testsC['Country'])



data = [trace1]

layout = dict(title = 'Testing Ratio versus Disease Prevalence',

              xaxis= dict(title= 'Disease Prevalence (Positive Tests/Million Population' ,ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Tests Done/ Positive Tests',ticklen= 5,zeroline= False)

             )



fig = dict(data = data, layout = layout)

iplot(fig)

                    
testsC['inverse_Test_Ratio']= testsC['Positive']/testsC['Tested']

testsC.sort_values(by=['inverse_Test_Ratio'], ascending=False, inplace=True)

trace1 = go.Bar(

                x = testsC['Country'],

                y = testsC['test_Ratio'],

                name = "Disease Prevalence/Testing Ratio",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = testsC['Country'])



data = [trace1]

             

layout = go.Layout(barmode = "group")



fig = go.Figure(data = data, layout = layout)

iplot(fig)

#  Define a variable "control_Metric" = disease prevalence/testing ratio = testsC['dp']/testsC['test_Ratio']

#  If disease prevalence is high and testing ratio is low, the z variable will be very high --> poor control

#  If disease prevalence is low and testing ratio is high, the z variable will be very low --> good control

testsC['control_Metric']= testsC['dp']/testsC['test_Ratio']



testsC.sort_values(by=['control_Metric'], ascending=True, inplace=True)

trace1 = go.Bar(

                x = testsC['Country'],

                y = testsC['control_Metric'],

                name = "Control Metric = Disease Prevalence/Testing Ratio",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = testsC['Country'])



data = [trace1]

             

layout = go.Layout(barmode = "group")



fig = go.Figure(data = data, layout = layout)

iplot(fig)

#  Try plotting the control_Metric versus disease prevalence

# Create first scatter plot for all countries

trace1 = go.Scatter(

                    x = testsC['dp'],

                    y = testsC['control_Metric'],

                    mode = "markers",

                    name = "Country",

                    marker = dict(color = 'rgba(255, 50, 50, 0.8)'),

                    text= testsC['Country'])



data = [trace1]

layout = dict(title = ' Control Metric versus Disease Prevalence',

              xaxis= dict(title= 'Disease Prevalence as positve tests/million population',ticklen= 5,zeroline= False),

              yaxis= dict(title= ' Control Metric defined as Disease Prevalence/Testing Ratio',ticklen= 5,zeroline= False)

             )



fig = dict(data = data, layout = layout)

iplot(fig)

                    