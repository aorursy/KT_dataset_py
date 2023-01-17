# load libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df.head()
df = df.rename(columns={'Country/Region' : 'Country'})

df = df.rename(columns={'ObservationDate' : 'Date'})

df.head()
df_countries = df.groupby(['Country','Date']).max().reset_index().sort_values('Date',ascending = False)

df_countries = df_countries.drop_duplicates('Country')

df_countries = df_countries[df_countries['Confirmed'] > 0]

df_countries
df_countrydate = df[df['Confirmed'] > 0]

df_countrydate = df_countrydate.groupby(['Date','Country']).sum().reset_index()

df_countrydate
fig = px.choropleth(df_countrydate, 

                    locations="Country", 

                    locationmode = "country names",

                    color="Confirmed", 

                    hover_name="Country", 

                    animation_frame="Date"

                   )



fig.update_layout(

    title_text = 'Spread of Coronavirus',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig.show()