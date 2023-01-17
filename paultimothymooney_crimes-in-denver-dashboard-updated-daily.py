import numpy as np

import pandas as pd 

import os

import time

from datetime import date, datetime

from dateutil import parser

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from scipy.special import boxcox

import plotly

import plotly.plotly as py

import plotly.offline as offline

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

from plotly.graph_objs import Scatter, Figure, Layout

cf.set_config_file(offline=True)

init_notebook_mode(connected=True)

%matplotlib inline



#

#import warnings

#warnings.simplefilter("ignore")

#



def CountWeekly(dataframe,column):

    dataframe['oldDate'] = dataframe[column]

    dataframe['newDate'] = pd.DatetimeIndex(dataframe.oldDate).normalize()

    dateCounts = dataframe['newDate'].value_counts()

    dateCountsDf = pd.DataFrame(dateCounts)

    dateCountsDf['date2'] = dateCountsDf.index

    dateCountsDf = dateCountsDf.sort_values(by='date2')

    dateCountsDf = dateCountsDf.resample('W', on='date2').sum()

    dateCountsDf = dateCountsDf.reset_index(level='date2')

    dateCountsDf = dateCountsDf[(dateCountsDf['date2'].dt.year > 2014)] 

    return dateCountsDf



def CountDaily(dataframe,column):

    dataframe['oldDate'] = dataframe[column]

    dataframe['newDate'] = pd.DatetimeIndex(dataframe.oldDate).normalize()

    dateCounts = dataframe['newDate'].value_counts()

    dateCountsDf = pd.DataFrame(dateCounts)

    dateCountsDf['date2'] = dateCountsDf.index

    dateCountsDf = dateCountsDf.sort_values(by='date2')

    dateCountsDf = dateCountsDf.resample('D', on='date2').sum()

    dateCountsDf = dateCountsDf.reset_index(level='date2')

    dateCountsDf = dateCountsDf[(dateCountsDf['date2'].dt.year > 2014)] 

    return dateCountsDf



crime = pd.read_csv('../input/crime.csv')
gtaCrimes = crime['OFFENSE_CATEGORY_ID']=='auto-theft'

gtaCrimes = crime[gtaCrimes]

gtaCrimes = CountWeekly(gtaCrimes,'REPORTED_DATE')

trace1 = go.Scatter(

                    x = gtaCrimes.date2,

                    y = gtaCrimes.newDate,

                    mode = "lines+markers",

                    name = "auto-theft",

                    marker = dict(color = 'black'),)



burglaryCrimes = crime['OFFENSE_CATEGORY_ID']=='burglary'

burglaryCrimes = crime[burglaryCrimes]

burglaryCrimes = CountWeekly(burglaryCrimes,'REPORTED_DATE')

trace2 = go.Scatter(

                    x = burglaryCrimes.date2,

                    y = burglaryCrimes.newDate,

                    mode = "lines+markers",

                    name = "burglary",

                    marker = dict(color = 'grey'),)



assaultCrimes = crime['OFFENSE_CATEGORY_ID']=='aggravated-assault'

assaultCrimes = crime[assaultCrimes]

assaultCrimes = CountWeekly(assaultCrimes,'REPORTED_DATE')

trace3 = go.Scatter(

                    x = assaultCrimes.date2,

                    y = assaultCrimes.newDate,

                    mode = "lines+markers",

                    name = "aggravated-assault",

                    marker = dict(color = 'blue'),)



alcoholCrimes = crime['OFFENSE_CATEGORY_ID']=='drug-alcohol'

alcoholCrimes = crime[alcoholCrimes]

alcoholCrimes = CountWeekly(alcoholCrimes,'REPORTED_DATE')

trace4 = go.Scatter(

                    x = alcoholCrimes.date2,

                    y = alcoholCrimes.newDate,

                    mode = "lines+markers",

                    name = "drug-alcohol",

                    marker = dict(color = 'green'),)



theftCrimes = crime['OFFENSE_CATEGORY_ID']=='theft-from-motor-vehicle'

theftCrimes = crime[theftCrimes]

theftCrimes = CountWeekly(theftCrimes,'REPORTED_DATE')

trace5 = go.Scatter(

                    x = theftCrimes.date2,

                    y = theftCrimes.newDate,

                    mode = "lines+markers",

                    name = "theft-from-motor-vehicle",

                    marker = dict(color = 'purple'),)





data = [trace1,trace2,trace3,trace4,trace5]

layout = dict(title = 'Weekly Count of Crimes in Denver',

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),yaxis= dict(title= '# of Incidents',ticklen= 5,zeroline= False),legend=dict(orientation= "h",x=0, y= 1.13)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
gtaCrimes = crime['OFFENSE_CATEGORY_ID']=='auto-theft'

gtaCrimes = crime[gtaCrimes]

gtaCrimes = CountDaily(gtaCrimes,'REPORTED_DATE')

trace1 = go.Scatter(

                    x = gtaCrimes.date2,

                    y = gtaCrimes.newDate,

                    mode = "lines+markers",

                    name = "auto-theft",

                    marker = dict(color = 'black',size=0.5),line=dict(width=0.5),)



burglaryCrimes = crime['OFFENSE_CATEGORY_ID']=='burglary'

burglaryCrimes = crime[burglaryCrimes]

burglaryCrimes = CountDaily(burglaryCrimes,'REPORTED_DATE')

trace2 = go.Scatter(

                    x = burglaryCrimes.date2,

                    y = burglaryCrimes.newDate,

                    mode = "lines+markers",

                    name = "burglary",

                    marker = dict(color = 'grey',size=0.5),line=dict(width=0.5),)



assaultCrimes = crime['OFFENSE_CATEGORY_ID']=='aggravated-assault'

assaultCrimes = crime[assaultCrimes]

assaultCrimes = CountDaily(assaultCrimes,'REPORTED_DATE')

trace3 = go.Scatter(

                    x = assaultCrimes.date2,

                    y = assaultCrimes.newDate,

                    mode = "lines+markers",

                    name = "aggravated-assault",

                    marker = dict(color = 'blue',size=0.5),line=dict(width=0.5),)



alcoholCrimes = crime['OFFENSE_CATEGORY_ID']=='drug-alcohol'

alcoholCrimes = crime[alcoholCrimes]

alcoholCrimes = CountDaily(alcoholCrimes,'REPORTED_DATE')

trace4 = go.Scatter(

                    x = alcoholCrimes.date2,

                    y = alcoholCrimes.newDate,

                    mode = "lines+markers",

                    name = "drug-alcohol",

                    marker = dict(color = 'green',size=0.5),line=dict(width=0.5),)



theftCrimes = crime['OFFENSE_CATEGORY_ID']=='theft-from-motor-vehicle'

theftCrimes = crime[theftCrimes]

theftCrimes = CountDaily(theftCrimes,'REPORTED_DATE')

trace5 = go.Scatter(

                    x = theftCrimes.date2,

                    y = theftCrimes.newDate,

                    mode = "lines+markers",

                    name = "theft-from-motor-vehicle",

                    marker = dict(color = 'purple',size=0.5),line=dict(width=0.5),)





data = [trace1,trace2,trace3,trace4,trace5]

layout = dict(title = 'Daily Count of Crimes in Denver',

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),yaxis= dict(title= '# of Incidents',ticklen= 5,zeroline= False,range=[0, 75]),legend=dict(orientation= "h",x=0, y= 1.13)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
counts = crime['OFFENSE_CATEGORY_ID'].value_counts()

counts = pd.DataFrame(counts)[0:10]

trace1 = go.Bar(

                x = counts.index,

                y = counts.OFFENSE_CATEGORY_ID,

                marker = dict(color = 'blue',

                             line=dict(color='black',width=1.5)),

                text = counts.OFFENSE_CATEGORY_ID)

data = [trace1]

layout = go.Layout(barmode = "group",title='Total Number of Crimes in the Denver Crime Dataset')

fig = go.Figure(data = data, layout = layout)

iplot(fig)