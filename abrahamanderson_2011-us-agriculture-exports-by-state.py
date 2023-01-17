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
#Geographical Plotting from various data:

#The choropleth maps of plotly is very useful afor geographical plotting:

#It is a thematic map in which areas are shaded in proportion to a statististical variable 

#that represents an aggregate summary of a geographic characteristics within each area, such as population density or GDP

import plotly as py

import plotly.graph_objs as go

import pandas as pd





from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot# these are functions we will use in plotly library

init_notebook_mode(connected=True) # here we initialize notebook mode
df=pd.read_csv("../input/ussimports.txt")

df.head()
#First we need to create a data dictionary because plotly only accepts certain types of data 

#as it is listed in this link: https://images.plot.ly/plotly-documentation/images/python_cheat_sheet.pdf

data = dict(type = 'choropleth', # this is type of math we select

            locations = df["code"],

            locationmode = 'USA-states', #here we select USA and its states,but there are different modes in the documentation

            colorscale= 'Blackbody', # there are different colors scales we can select like 

#Other type of available colorscales:'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

             

            text= df["state"], # this is information and what hovers over each of locations in the map

            z=df["total exports"], #it is actual values that will be shown in the colorscale.

            colorbar = {'title':'Millions USD'})
# we also need to create a layout object of plotly:

# next we need to create a layout object:

layout=dict(geo={"scope":"usa"})
go.Figure(data,layout)# this is the function we call the graph of the map

# here we pass the two necessary object into figure function