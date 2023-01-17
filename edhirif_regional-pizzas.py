# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/8358_1.csv') #get the data
df_unique = df.drop_duplicates('id') 
df_unique.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4)
df_type = df_unique['menus.name'].value_counts()

print(df_type.head(20))
df_ps = df_unique[df_unique['menus.name']=='Pizza Steak'] #new dataframe just for pizza steak



data = [ dict(

        type = 'scattergeo',

        locationmode = 'USA-states',

        lon = df_ps['longitude'],

        lat = df_ps['latitude'],

        #text = df['text'],

        mode = 'markers',

        marker = dict( 

            size = 8, 

            opacity = 0.6,

            reversescale = True,

            autocolorscale = False,

            symbol = 'circle',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            )))]

            

layout = dict(

        title = 'location of pizza steak',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            subunitcolor = "rgb(217, 217, 217)",

            countrycolor = "rgb(217, 217, 217)",

            countrywidth = 0.5,

            subunitwidth = 0.5))

import plotly.offline as py

py.init_notebook_mode(connected=True)

fig = dict( data=data, layout=layout )

py.iplot(fig, filename='pizzasteak.html')
df_wp = df_unique[df_unique['menus.name']=='White Pizza']

data = [ dict(

        type = 'scattergeo',

        locationmode = 'USA-states',

        lon = df_wp['longitude'],

        lat = df_wp['latitude'],

        #text = df['text'],

        mode = 'markers',

        marker = dict( 

            size = 8, 

            opacity = 0.6,

            reversescale = True,

            autocolorscale = False,

            symbol = 'circle',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            )))]

            

layout = dict(

        title = 'location of white pizza',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            subunitcolor = "rgb(217, 217, 217)",

            countrycolor = "rgb(217, 217, 217)",

            countrywidth = 0.5,

            subunitwidth = 0.5))

import plotly.offline as py

py.init_notebook_mode(connected=True)

fig = dict( data=data, layout=layout )

py.iplot(fig, filename='whitepizza.html')
