



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex2 import *

print("Setup Complete")
import pandas as pd

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



covid_19_India = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")

IndividualDetails = pd.read_csv("../input/covid19-in-india/IndividualDetails.csv")

HospitalBedsIndia = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")

ICMRTestingDetails = pd.read_csv("../input/covid19-in-india/ICMRTestingDetails.csv")

population_india_census2011 = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")



covid_19_India.columns
features=['Sno', 'Date', 'State/UnionTerritory', 'ConfirmedIndianNational',

       'ConfirmedForeignNational', 'Cured', 'Deaths']

covid_19_India=covid_19_India[features]

display(covid_19_India)
import numpy as np 

import pandas as pd

from datetime import date

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Markdown

import plotly.graph_objs as go

import plotly.offline as py

from plotly.subplots import make_subplots

import plotly.express as px

from plotly.offline import init_notebook_mode, plot, iplot, download_plotlyjs

import plotly as ply

import pycountry

import folium 

from folium import plugins

import json





%config InlineBackend.figure_format = 'retina'

init_notebook_mode(connected=True)



# color pallette

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active case - yellow



# India Latitude Longitude

India_Latitude = 21.7679

India_Longitude = 78.8718 
# Utility Functions



'''Display markdown formatted output like bold, italic bold etc.'''

def formatted_text(string):

    display(Markdown(string))





'''highlight the maximum in a Series or DataFrame'''  

def highlight_max(data, color='red'):

    attr = 'background-color: {}'.format(color)

    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1

        is_max = data == data.max()

        return [attr if v else '' for v in is_max]

    else:  # from .apply(axis=None)

        is_max = data == data.max().max()

        return pd.DataFrame(np.where(is_max, attr, ''), index=data.index, columns=data.columns)

    





# Utility Plotting Functions



def plotDailyReportedCasesOverTime(df, country):

    # confirmed

    fig = px.bar(df, x="Date", y="Confirmed")

    layout = go.Layout(

 title=go.layout.Title(text="Daily count of confirmed cases in "+ country, x=0.5),

        font=dict(size=14),

        width=800,

        height=500,

        xaxis_title = "Date",

        yaxis_title = "Confirmed cases")



    fig.update_layout(layout)

    fig.show()



    # deaths

    fig = px.bar(df, x="Date", y="Deaths")

    layout = go.Layout(

        title=go.layout.Title(text="Daily count of reported deaths in "+ country, x=0.5),

        font=dict(size=14),

        width=800,

        height=500,

        xaxis_title = "Date",

        yaxis_title = "Deaths Reported")



    fig.update_layout(layout)

    fig.show()



    # recovered

    fig = px.bar(df, x="Date", y="Recovered")

    layout = go.Layout(

        title=go.layout.Title(text="Daily count of recovered cases in "+ country, x=0.5),

        font=dict(size=14),

        width=800,

        height=500,

        xaxis_title = "Date",

        yaxis_title = "Recovered Cases")



    fig.update_layout(layout)

    fig.show()

    

# Cases over time

def scatterPlotCasesOverTime(df, country):

    plot = make_subplots(rows=1, cols=3, subplot_titles=("Comfirmed", "Deaths", "Recovered"))



    subPlot1 = go.Scatter(

                    x=df['Date'],

                    y=df['Confirmed'],

                    name="Confirmed",

                    line_color='orange',

        opacity=0.8)



    subPlot2 = go.Scatter(

                    x=df['Date'],

                    y=df['Deaths'],

                    name="Deaths",

                    line_color='red',

                    opacity=0.8)



    subPlot3 = go.Scatter(

                    x=df['Date'],

                    y=df['Recovered'],

                    name="Recovered",

                    line_color='green',

                    opacity=0.8)



    plot.append_trace(subPlot1, 1, 1)

    plot.append_trace(subPlot2, 1, 2)

    plot.append_trace(subPlot3, 1, 3)

    plot.update_layout(template="ggplot2", title_text = country + '<b> - Spread of the nCov Over Time</b>')



    plot.show()
covid_19_India['Confirmed'] = covid_19_India['ConfirmedIndianNational'] + covid_19_India['ConfirmedForeignNational']

covid_19_India.rename(columns={'State/UnionTerritory': 'State'}, inplace=True)

display(covid_19_India)
statewise_cases = pd.DataFrame(covid_19_India.groupby(['State'])['Confirmed', 'Deaths', 'Cured'].max().reset_index())

statewise_cases["Country"] = "India" # in order to have a single root node

fig = px.bar(statewise_cases,x="Confirmed", y="State", title='Confirmed Cases', text='Confirmed', orientation='h', 

             width=16*(max(statewise_cases['Confirmed']) + 2), height=700, range_x = [0, max(statewise_cases['Confirmed']) + 2])

fig.update_traces(marker_color='#0726ed', opacity=0.8, textposition='outside')



fig.update_layout(plot_bgcolor='rgb(208, 236, 245)')

fig.show()
date_wise_data = covid_19_India[["Date","Confirmed","Deaths","Cured"]]

date_wise_data['Date'] = date_wise_data['Date'].apply(pd.to_datetime, dayfirst=True)

date_wise_data
date_wise_data = date_wise_data.groupby(["Date"]).sum().reset_index()



formatted_text('***Date wise data***')

date_wise_data
temp = date_wise_data.melt(id_vars="Date", value_vars=['Cured', 'Deaths', 'Confirmed'],

                 var_name='Case', value_name='Count')



fig = px.area(temp, x="Date", y="Count", color='Case',title='Time wise cases analysis', color_discrete_sequence = [rec, dth, act])

fig.show()



statewise_cases = pd.DataFrame(covid_19_India.groupby(['State'])['Confirmed', 'Deaths', 'Cured'].max().reset_index())

statewise_cases["Country"] = "India" # in order to have a single root node

fig = px.treemap(statewise_cases, path=['Country','State'], values='Confirmed',

                  color='Confirmed', hover_data=['State'],

                  color_continuous_scale='RdBu')

fig.show()