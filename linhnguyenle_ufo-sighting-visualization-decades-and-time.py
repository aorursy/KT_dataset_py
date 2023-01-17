# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib as lib

import pandas as pd

from bs4 import BeautifulSoup

import requests

import json

import numpy as np
data=pd.read_csv('../input/scrubbed.csv',dtype={"duration (seconds)": str,'latitude':str,'shape':str,'comments':str,'city':str})
us=data[data.country=='us']
import re

def getDuration(row):

    return float(re.findall('\d+', row['duration (seconds)'].strip('`') )[0])

us['duration']=us.apply(getDuration,axis=1)
def getLatitude(row):

    return float(row['latitude'].replace('p',''))

us['latitudeN']=us.apply(getLatitude,axis=1)
import string

def getRightDuration(row):

    x=row['duration']

    if x>15768000:

        return str(x//15768000)+' years'

    elif x>1296000:

        return str(x//1296000)+' months and '+str((x%1296000)//(60*60*12))+' days'

    elif x>302400:

        return str(x//302400)+' weeks and '+str(x%302400//(60*60*12))+' days'

    elif x>43200:

        return str(x//43200)+' days and '+str(x%43200//(12))+' hours'

    elif x>3600:

        return str(x//3600)+' hours and '+str(x%3600//60)+ 'minutes'

    elif x>60:

        return str(x//60)+' minutes'

    else:

        return str(x)+' seconds'

def formatName(row):

    city=row['city'].split()

    name=''

    for i in city:

        name+=i[0].upper()+i[1:]+' '

    return name
def abduct(row):

    if type(row['comments'])!=str:

        return 0

    if 'abduct' in row['comments']:

        return 1

    else:

        return 0
def getShape(row):

    if type(row['shape']!=str):

        return 'unknown'

    elif row['shape']=='nan' or row['shape']=='NaN':

        return 'unknown'

    else:

        return str(row['shape'])
def military(row):

    if type(row['comments'])!=str:

        return 0

    if 'military' in row['comments'] or 'army' in row['comments'] or 'aircraft' in row['comments'] or 'air craft' in row['comments']:

        return 1

    else:

        return 0
useD=us[us.duration>900]

useD['state']=useD['state'].map(lambda x: x.upper())

useD['city']=useD['city'].replace('NaN','unknown')

useD['abduct']=useD.apply(abduct,axis=1)

useD['durationStr']=useD.apply(getRightDuration,axis=1)

useD['city']=useD.apply(formatName,axis=1)

useD['shape']=useD.apply(getShape,axis=1)

useD['military']=useD.apply(military,axis=1)

useD.head()
useD=useD.sort_values('duration',ascending=False).reset_index()
import plotly.offline as py

py.offline.init_notebook_mode()
def getDecade(row):

    x=int(row['datetime'].split('/')[2].split(' ')[0])

    return x-(x%10)
useD['decade']=useD.apply(getDecade,axis=1)
def getTime(row):

    x= row['datetime'].split('/')[2].split(' ')[1].split(':')[0]

    return x+':00 - '+x+':59'
useD['time']=useD.apply(getTime,axis=1)
from ipywidgets import interact

%matplotlib inline

import matplotlib.pyplot as plt

import networkx as nx
from ipywidgets import interact, interactive, fixed

import ipywidgets as widgets
def drawDecade(decade):

    useD['text'] = useD['city']+','+useD['state'] + '<br>Duration: ' +useD['durationStr']+'<br>Shape: '+useD['shape']

    # limits = [(0,34),(35,183),(184,2416),(2417,5714),(5715,11163)]

    #legend=['1930','1940','1950','1960','1970','1980','1990','2000','2010']

    #colors = ["red","orange","blue","green","lightgrey",'yellow','pink','black','purple','grey']

    cities = []

    scale = 50000

#     for i in range(len(legend)):

    df_sub = useD[useD.decade==int(decade)]

    city = dict(

        type = 'scattergeo',

        locationmode = 'USA-states',

        lon = df_sub['longitude '],

        lat = df_sub['latitude'],

        text=df_sub['text'],

        marker = dict(

#             size = df_sub['duration']/scale,

            color = 'red',

            line = dict(width=0.5, color='rgb(40,40,40)'),

            sizemode = 'area'

        ),

        name = str(decade)+'s' )

    cities.append(city)







    layout = dict(

            title = 'UFO Sighting according to Decade ',

            showlegend = True,

            geo = dict(

                scope='usa',

                projection=dict( type='albers usa' ),

                showland = True,

                landcolor = 'rgb(217, 217, 217)',

                subunitwidth=1,

                countrywidth=1,

                subunitcolor="rgb(255, 255, 255)",

                countrycolor="rgb(255, 255, 255)"

            )

        )



    fig1 = dict(data=cities, layout=layout)

    py.iplot( fig1, validate=True)
def drawTime(time):

    useD['text'] = useD['city']+','+useD['state'] + '<br>Duration: ' +useD['durationStr']+'<br>Shape: '+useD['shape']

    # limits = [(0,34),(35,183),(184,2416),(2417,5714),(5715,11163)]

    #legend=['1930','1940','1950','1960','1970','1980','1990','2000','2010']

    #colors = ["red","orange","blue","green","lightgrey",'yellow','pink','black','purple','grey']

    cities = []

    scale = 50000

#     for i in range(len(legend)):

    df_sub = useD[useD.time==str(time)]

    city = dict(

        type = 'scattergeo',

        locationmode = 'USA-states',

        lon = df_sub['longitude '],

        lat = df_sub['latitude'],

        text=df_sub['text'],

        marker = dict(

#             size = df_sub['duration']/scale,

            color = 'blue',

            line = dict(width=0.5, color='rgb(40,40,40)'),

            sizemode = 'area'

        ),

        name = str(time) )

    cities.append(city)







    layout = dict(

            title = 'UFO Sighting according to Time of the Day',

            showlegend = True,

            geo = dict(

                scope='usa',

                projection=dict( type='albers usa' ),

                showland = True,

                landcolor = 'rgb(217, 217, 217)',

                subunitwidth=1,

                countrywidth=1,

                subunitcolor="rgb(255, 255, 255)",

                countrycolor="rgb(255, 255, 255)"

            )

        )



    fig1 = dict(data=cities, layout=layout)

    py.iplot( fig1, validate=True)
from collections import OrderedDict

def drawDecades(Decade):

    return drawDecade(Decade)

interact(drawDecades,Decade=OrderedDict([('1930s', 1930),

                                     ('1940s',1940),

                                     ('1950s',1950),

                                     ('1960s',1960),

                                      ('1970s',1970),

                                    ('1980s',1980),

                                       ('1990s',1990),

                                        ('2000s',2000),

                                         ('2010s',2010)]))

x={}

for y in useD['time'].unique():

    x[y]=(y)

def getTime(Time):

    return drawTime(Time)

interact(getTime,Time=OrderedDict((time,time) for time in sorted(x.keys())))