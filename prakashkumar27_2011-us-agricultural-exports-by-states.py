# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Connect Data
init_notebook_mode(connected=True) 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read US Agriculture Export data file
df=  pd.read_csv("../input/2011_US_AGRI_Exports")
df.head()
# Create data using dictionary for the Choropleth arguments
data = dict(type='choropleth',
            colorscale = 'YlOrRd',
            locations = df['code'],
            z = df['total exports'],
            locationmode = 'USA-states',
            text = df['text'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"Millions USD"}
            ) 
#Make layout using dictionary with the arguments
layout = dict(title = '2011 US Agriculture Exports by State',
              geo = dict(scope='usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)')
             )

#Create choromap using data and layout
choromap = go.Figure(data = [data],layout = layout)
# Displaying the Result Choropleth Map using iplot
# Each state with the total exports in millions in USD of the agricultural products
iplot(choromap)