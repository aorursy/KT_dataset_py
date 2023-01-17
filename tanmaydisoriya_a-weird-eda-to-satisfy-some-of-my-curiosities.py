# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import seaborn as sns
%matplotlib inline
#I think these things would have been done better on imdb database but, anyway 
data = pd.read_csv("../input/wiki_movie_plots_deduped.csv")
data.head()
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import plotly.graph_objs as go


temp=data['Release Year'].value_counts()
temp=temp.sort_index(ascending=True)
Year=temp.index
Number=temp.values
trace = [go.Scatter(
    x = Year,
    y = Number,
    mode = 'lines',
    name = 'lines'
)]
layout = go.Layout(
    title='Number of movies by year',
)
fig = go.Figure(data=trace, layout=layout)
iplot(fig)
data['Origin/Ethnicity'].unique()
data.groupby( [ "Origin/Ethnicity"])['Title'].count().sort_values(ascending=False)

Indian = ['Assamese','Bengali','Bollywood','Kannada','Malayalam','Marathi','Punjabi', 
          'Tamil', 'Telugu']
Others = ['Australian', 'Bangladeshi','Egyptian', 'Hong Kong', 'Filipino'
          , 'Malaysian', 'Maldivian',
           'Russian', 'South_Korean', 'Turkish']

Indian_data = data[data['Origin/Ethnicity'].isin(Indian)]
Others_data = data[data['Origin/Ethnicity'].isin(Others)]
American_data=data[data['Origin/Ethnicity']=='American']
British_data=data[data['Origin/Ethnicity']=='British']
Canadian_data=data[data['Origin/Ethnicity']=='Canadian']
Chinese_data=data[data['Origin/Ethnicity']=='Chinese']
Japanese_data=data[data['Origin/Ethnicity']=='Japanese']
temp=Indian_data['Release Year'].value_counts().sort_index(ascending=True)
trace1 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'Indian'
)
temp=Others_data['Release Year'].value_counts().sort_index(ascending=True)
trace2 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'Others'
)
temp=American_data['Release Year'].value_counts().sort_index(ascending=True)
trace3 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'American'
)
temp=British_data['Release Year'].value_counts().sort_index(ascending=True)
trace4 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'British'
)
temp=Chinese_data['Release Year'].value_counts().sort_index(ascending=True)
trace5 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'Chinese'
)
temp=Canadian_data['Release Year'].value_counts().sort_index(ascending=True)
trace6 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'Canadian'
)
temp=Japanese_data['Release Year'].value_counts().sort_index(ascending=True)
trace7 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'Japanese'
)
layout = go.Layout(
    title='Number of movies classified by origin per year',
)

datapoints = [trace1, trace2,trace3,trace4,trace5,trace6,trace7]
fig = dict(data=datapoints,layout=layout)
iplot(fig)
data.groupby( [ "Genre"])['Title'].count().sort_values(ascending=False)
temp_data=data[data['Genre']=='drama']
temp=temp_data['Release Year'].value_counts().sort_index(ascending=True)
trace1 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'drama'
)
temp_data=data[data['Genre']=='comedy']
temp=temp_data['Release Year'].value_counts().sort_index(ascending=True)
trace2 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'comedy'
)

temp_data=data[data['Genre']=='horror']
temp=temp_data['Release Year'].value_counts().sort_index(ascending=True)
trace3 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'horror'
)

temp_data=data[data['Genre']=='action']
temp=temp_data['Release Year'].value_counts().sort_index(ascending=True)
trace4 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'action'
)

temp_data=data[data['Genre']=='thriller']
temp=temp_data['Release Year'].value_counts().sort_index(ascending=True)
trace5 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'thriller'
)

temp_data=data[data['Genre']=='romance']
temp=temp_data['Release Year'].value_counts().sort_index(ascending=True)
trace6 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'romance'
)

temp_data=data[data['Genre']=='western']
temp=temp_data['Release Year'].value_counts().sort_index(ascending=True)
trace7 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'western'
)


temp_data=data[data['Genre']=='film noir']
temp=temp_data['Release Year'].value_counts().sort_index(ascending=True)
trace8 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'film noir'
)

layout = go.Layout(
    title='Number of movies classified by Genre per year',
    
)

datapoints = [trace1, trace2,trace3,trace4,trace5,trace6,trace7,trace8]
fig = dict(data=datapoints,layout=layout)
iplot(fig)
##total=data['Release Year'].value_counts().sort_index(ascending=True).values
##total
layout = go.Layout(
    title='Film Noir by year',
    annotations= [
    {
      "x": 1945, 
      "y": 1.06818181818, 
      "font": {
        "color": "rgb(255, 0, 0)", 
        "size": 8
      }, 
      "showarrow": False, 
      "text": "Rise in popularity of Film Noir", 
      "xref": "x", 
      "yref": "paper"
    }],
    shapes= [
        {
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper',
            'x0': '1940',
            'y0': 0,
            'x1': '1950',
            'y1': 1,
            'fillcolor': '#d3d3d3',
            'opacity': 0.5,
            'line': {
                'width': 0,
            }
        }]
)



datapoints = [trace8]
fig = dict(data=datapoints,layout=layout)
iplot(fig)

temp=data['Release Year'].value_counts()
temp=temp.sort_index(ascending=True)
Year=temp.index
Number=temp.values
trace0 = go.Scatter(
    x = Year,
    y = Number,
    mode = 'lines',
    name = 'Total'
)

temp=American_data['Release Year'].value_counts().sort_index(ascending=True)
trace1 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'American'
)

layout = go.Layout(
    title='Number of movies by year',
    annotations= [
    {
      "x": 2007, 
      "y": 1, 
      "font": {
        "color": "rgb(255, 0, 0)", 
        "size": 8
      }, 
      "showarrow": True, 
      "text": "2007 Financial crysis", 
      "xref": "x", 
      "yref": "paper"
    },
    {
      "x": 1942, 
      "y": 1, 
      "font": {
        "color": "rgb(255, 0, 0)", 
        "size": 8
      }, 
      "showarrow": False, 
      "text": "WWII", 
      "xref": "x", 
      "yref": "paper"
    },
    {
      "x": 1916, 
      "y": 1, 
      "font": {
        "color": "rgb(255, 0, 0)", 
        "size": 8
      }, 
      "showarrow": False, 
      "text": "WWI", 
      "xref": "x", 
      "yref": "paper"
    }],
    shapes= [
        {
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper',
            'x0': '1939',
            'y0': 0,
            'x1': '1945',
            'y1': 1,
            'fillcolor': '#d3d3d3',
            'opacity': 0.5,
            'line': {
                'width': 0,
            }
        },
         {
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper',
            'x0': '2007',
            'y0': 0,
            'x1': '2007.4',
            'y1': 1,
            'fillcolor': '#d3d3d3',
            'opacity': 1,
            'line': {
                'width': 0,
            }
        },
        {
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper',
            'x0': '1914',
            'y0': 0,
            'x1': '1918',
            'y1': 1,
            'fillcolor': '#d3d3d3',
            'opacity': 0.5,
            'line': {
                'width': 0,
            }
        }
        ]
)

fig = go.Figure(data=[trace0,trace1], layout=layout)
iplot(fig)
Indian_Pop = [449480608,458494963,467852537,477527970,487484535,497702365,508161935,518889779,529967317,541505076,553578513,566224812,579411513,593058926,607050255,621301720,635771734,650485030,665502284,680915804,696783517,713118032,729868013,746949067,764245202,781666671,799181436,816792741,834489322,852270034,870133480,888054875,906021106,924057817,942204249,960482795,978893217,997405318,1015974042,1034539214,1053050912,1071477855,1089807112,1108027848,1126135777,1144118674,1161977719,1179681239,1197146906,1214270132,1230980691,1247236029,1263065852,1278562207,1293859294,1309053980,1324171354,1339180127]
Indian_Pop_Million = np.array(Indian_Pop)/1000000
Indian_GDP = [36535925031.34,38709096075.6305,41599070242.3094,47776000900.0302,55726873083.5543,58760424669.8482,45253641303.1897,49466168890.9507,52377324284.1951,57668330026.3629,61589800519.5084,66452561865.8332,70509913049.4003,84374541630.2062,98198276856.6209,97159222024.1364,101346972433.934,119866746574.408,135468782808.69,150950826964.424,183839864649.15,190909548789.769,198037712681.605,215350771428.331,209328156800.867,229410293759.071,245664654062.873,275311425331.64,292632656262.687,292093308319.642,316697337894.513,266502281094.117,284363884080.101,275570363431.902,322909902308.892,355475984177.451,387656017798.596,410320300470.283,415730874171.13,452699998386.914,462146799337.698,478965491060.771,508068952065.901,599592902016.345,699688852930.276,808901077222.839,920316529729.747,1201111768410.27,1186952757636.11,1323940295874.06,1656617073124.71,1823049927772.05,1827637859136.23,1856722121394.42,2039127446299.3,2102390808997.09,2274229710530.03,2597491162897.67
        ]
Indian_GDP_Million = np.array(Indian_GDP)/1000000000
Year = [1960,1961,1962,1963,1964,1965,1966,1967,1968,1969,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
temp=Indian_data['Release Year'].value_counts().sort_index(ascending=True)
trace3 = go.Scatter(
    x = Year,
    y = Indian_GDP_Million,
    mode = 'lines',
    name = 'Indian_GDP In Billion USD'
)
trace2 = go.Scatter(
    x = Year,
    
    y = Indian_Pop_Million,
    mode = 'lines',
    name = 'Indian_Population In Million'
)
trace1 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'Indian'
)

datapoints = [trace1,trace2,trace3]
fig = dict(data=datapoints)
iplot(fig)
data['Soviet']=data.Plot.str.contains('Soviet', case=False)
American_data=data[data['Origin/Ethnicity']=='American']
temp_data=American_data[data['Soviet']==True]
temp=temp_data['Release Year'].value_counts().sort_index(ascending=True)
trace1 = go.Scatter(
    x = temp.index,
    y = temp.values,
    mode = 'lines',
    name = 'Instances of Occurrence '
)
layout = go.Layout(
    title='Number of movies wit  by year',
    annotations= [
    {
      "x": 1991, 
      "y": 1, 
      "font": {
        "color": "rgb(255, 0, 0)", 
        "size": 8
      }, 
      "showarrow": True, 
      "text": "Dissolution of the Soviet Union", 
      "xref": "x", 
      "yref": "paper"
    }
        
    ],
    shapes= [

         {
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper',
            'x0': '1991',
            'y0': 0,
            'x1': '1991.4',
            'y1': 1,
            'fillcolor': '#d3d3d3',
            'opacity': 1,
            'line': {
                'width': 0,
            }
        }
        
        ]
)


datapoints = [trace1]
fig = dict(data=datapoints,layout=layout)
iplot(fig)
