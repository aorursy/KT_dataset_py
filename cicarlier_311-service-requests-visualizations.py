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
df = pd.read_csv("../input/311-service-requests-abandoned-vehicles.csv")
df.head()
df['Completion Date'] = pd.to_datetime(df['Completion Date'],format='%Y-%m-%dT%H:%M:%S')
df['Creation Date'] = pd.to_datetime(df['Creation Date'],format='%Y-%m-%dT%H:%M:%S')
df.head()
df['Time to complete'] = df['Completion Date'] - df['Creation Date']
df.head()
df1 = df.groupby(['ZIP Code', 'Vehicle Make/Model'])['ZIP Code'].count().unstack()
df1
import matplotlib.pyplot as plt
df.groupby('ZIP Code')['Creation Date'].count().plot(kind='bar')
plt.show()
plt.clf()
df['Creation Date'].map(lambda d: d.month).plot(kind='hist')
plt.show()
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
trace0 = go.Scatter(
    x = month,
    y = df['Creation Date'][df['Creation Date'].dt.year == 2011].groupby([df['Creation Date'].dt.year, df['Creation Date'].dt.month]).agg('count'),
    mode = 'lines',
    name = '2011'
)
trace1 = go.Scatter(
    x = month,
    y = df['Creation Date'][df['Creation Date'].dt.year == 2012].groupby([df['Creation Date'].dt.year, df['Creation Date'].dt.month]).agg('count'),
    mode = 'lines',
    name = '2012'
)
trace2 = go.Scatter(
    x = month,
    y = df['Creation Date'][df['Creation Date'].dt.year == 2013].groupby([df['Creation Date'].dt.year, df['Creation Date'].dt.month]).agg('count'),
    mode = 'lines',
    name = '2013'
)
trace3 = go.Scatter(
    x = month,
    y = df['Creation Date'][df['Creation Date'].dt.year == 2014].groupby([df['Creation Date'].dt.year, df['Creation Date'].dt.month]).agg('count'),
    mode = 'lines',
    name = '2014'
)
data = [trace0, trace1, trace2, trace3]

# specify the layout of our figure
layout = dict(title = "Number of Requests",
              xaxis= dict(title= 'Month',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)