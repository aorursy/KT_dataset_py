# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go #plotly libraries for plotting figures

import plotly.express as px #plotly express





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#try first with LFB Incident data from January 2017 mini.xlsx which contains only 1000 rows

df = pd.read_excel('/kaggle/input/londonfirebrigadeincidentsrecordsmini/LFB Incident data from January 2017 mini.xlsx')
#running this cell with the entire dataset will take a long time because of the number of rows

#df = pd.read_excel('/kaggle/input/londonfirebrigadeincidentrecords/LFB Incident data from January 2017.xlsx')
#this plots the first 5 rows, just to understand the columns in the dataset and have a quick overview of the data

df.head()
#this plots a list of the name of the columns so that we can easily copy/past their name for calling them in subsequent cells

df.columns
#drop any nan values from dataframe df

#how='any' means we are dropping the entire row if just one of the values is nan

#refer to https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html for documentation

df=df.dropna(how='any')
#lets see what we get after we have dropped all nan values

#describe gives us an idea of the dataset, including some stats

df.describe()
#and now if we plot the first three rows head(3) = first 3 rows

df.head(3)
#note that the first column is not sequential, we can reindex usint pd.reset_index if needed 

#see documentation here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html?highlight=reset_index#pandas.DataFrame.reset_index

#We can use the drop parameter (drop=True) to avoid the old index being added as a column

#if you would like to keep the original indexes, then either do not run this cell or comment the following line

#df.reset_index(drop=True)
mapbox_access_token='pk.eyJ1IjoibWFyaWEtbWluZ2FsbG9uIiwiYSI6ImNrYzhxMHppMjFlNjUyeXFvaHBpODhjM20ifQ.gPWY1eHpfN23yVIi2qc7FA'
import plotly.express as px



px.set_mapbox_access_token(mapbox_access_token)



fig = px.scatter_mapbox(df, 

                        lat=df['Latitude'], 

                        lon=df['Longitude'],     

                        color=df['IncidentStationGround'], 

                        size=df['Notional Cost (£)'],

                        size_max=50,

                        color_continuous_scale=px.colors.cyclical.IceFire,

                        animation_frame=df['CalYear'],

                        center= dict(lat=51.515419,lon=-0.141099),

                        zoom=9,

                        width=1080,

                        height=900,

                        title = "London Fire Brigade Records",

                        mapbox_style="basic")

fig.show()