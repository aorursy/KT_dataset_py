# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

py.init_notebook_mode(connected = True)
data=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles_nov_2019.csv')
data.head()
data.info()
data.dtypes
data.rating.value_counts()
def pie_plot(cnt_srs, colors, title):

    labels=cnt_srs.index

    values=cnt_srs.values

    trace = go.Pie(labels=labels, 

                   values=values, 

                   title=title, 

                   hoverinfo='percent+value', 

                   textinfo='percent',

                   textposition='inside',

                   hole=0.7,

                   showlegend=True,

                   marker=dict(colors=colors,

                               line=dict(color='#000000',

                                         width=2),

                              )

                  )

    return trace



py.iplot([pie_plot(data['type'].value_counts(), ['green', 'red'], 'Type')])
def pie_plot(cnt_srs, title):

    labels=cnt_srs.index

    values=cnt_srs.values

    trace = go.Pie(labels=labels, 

                   values=values, 

                   title=title, 

                   hoverinfo='percent+value', 

                   textinfo='percent',

                   showlegend=True,

                   marker=dict(colors=plt.cm.viridis_r(np.linspace(0, 1, 14)),

                               line=dict(color='#000000',

                                         width=1.5),

                              )

                  )

    return trace



py.iplot([pie_plot(data['rating'].value_counts(), 'Type')])
temp_data = data['rating'].value_counts().reset_index()

graph1 = go.Bar(

                x = temp_data['index'],

                y = temp_data['rating'],

                marker = dict(color = 'rgb(204,0,0)',

                              line=dict(color='rgb(0,0,0)',width=2.0)))

layout = go.Layout(template= "plotly_white",title = 'GRAPH BASED ON RATING' , 

                   xaxis = dict(title = 'Rating'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [graph1], layout = layout)

fig.show()
data.rating.describe()