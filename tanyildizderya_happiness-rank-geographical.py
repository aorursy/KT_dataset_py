# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import plotly.plotly as py 

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
df = pd.read_csv('../input/2015.csv')
df.head()
data = dict(type='choropleth',

           locations = df['Country'],

           locationmode='country names',

           z = df['Happiness Score'],

           text = df['Country'],

           colorbar= {'title' : 'Happiness Score'})
layout = dict(title='2015 Happiness Rank',

            geo = dict(showframe = False,

             projection = {'type' : 'kavrayskiy7'}))
choromap = go.Figure(data=[data],layout = layout)
iplot(choromap)
i