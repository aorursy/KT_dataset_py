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













import plotly.plotly as py

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import plotly.offline as po

po.init_notebook_mode(connected=True)

df = pd.read_csv('../input/population.csv')

df.head()



df['count']=0
df.head()


for i in range(len(df)):

    if (df.iloc[i,2]>df.iloc[i,3]):

        df.iloc[i,5]=1

else:

    df.iloc[i,5]=0
df
import matplotlib.cm as cm

import plotly.offline as po

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

data = dict(type='choropleth',

locations = df['name'],

locationmode = 'country names', z = df['count'],

text = df['name'], colorbar = dict(title='Frequency'),

colorscale=[[0,"rgb(255,0, 0)"],[1,"rgb(0,0,255)"]],

autocolorscale = False, showscale = False,

          

           )
layout = dict(title='Country color as blue if it has more female population than male else color is red',              

geo = dict(showframe = False, projection={'type':'equirectangular'},showlakes = False,

        showcoastlines = True,showland = True,

        landcolor = "rgb(229,229,229)"

             ), showlegend=True)
choromap = go.Figure(data = [data], layout = layout)

iplot(choromap, validate=False)

plt.show()