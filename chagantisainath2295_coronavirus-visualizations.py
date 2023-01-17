# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing libraries

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots 

from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)
#Reading data

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
#Checking columns

df.head(n=10)
df.plot('Deaths')
#Rename columns 

df = df.rename(columns={'Country/Region':'Country'})

df=df.rename(columns={'ObservationDate':'Date'})

df.head()
#Manipulate Dataframe



df_countries = df.groupby(['Country','Date']).sum().reset_index().sort_values('Date',ascending = False)

df_countries = df_countries.drop_duplicates(subset=['Country'])

df_countries = df_countries[df_countries['Confirmed']>0]
#Creating Choropleth

fig = go.Figure(data=go.Choropleth

                (locations=df_countries['Country'],

                 locationmode = 'country names',z=df_countries['Confirmed']

                 ,colorscale='Reds',marker_line_color = 'black',marker_line_width = 0.5,

 ))



fig.update_layout(

title_text = 'Confirmed Cases as of March 28,2020',

title_x = 0.5,

geo= dict(

     showframe = False,

    showcoastlines = False,

projection_type = 'equirectangular'

 )

)
# Manipulating the original dataframe

df_countrydate = df[df['Confirmed']>0]

df_countrydate = df_countrydate.groupby(['Date','Country']).sum().reset_index()

df_countrydate

# Creating the visualization

fig = px.choropleth(df_countrydate, 

                    locations="Country", 

                    locationmode = "country names",

                    color="Confirmed", 

                    hover_name="Country", 

                    animation_frame="Date"

                   )

fig.update_layout(

    title_text = 'Global Spread of Coronavirus',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig.show()