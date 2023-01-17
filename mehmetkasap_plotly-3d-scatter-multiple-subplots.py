# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

# plotly library
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# matplotlib library
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
timesData = pd.read_csv('../input/timesData.csv')
timesData.info()
dataframe = timesData[timesData.year == 2015]
dataframe.head()
trace1 = go.Scatter3d(x = dataframe.world_rank,
                      y = dataframe.research,
                      z = dataframe.citations,
                      mode = 'markers',
                      marker = dict(size = 10, color = dataframe.research)
                      )
# in trace1, color changes according to the research
# we can also arrange size values as color, for instance, we can set size = dataframe.citations

data = [trace1]
layout = go.Layout(margin = dict(l=0, r=0, b=0, t=0))
fig = go.Figure(data=data, layout=layout)
iplot(fig)
plt.savefig('3D_Scatter.png')
dataframe.columns
# first we create 4 traces
trace1 = go.Scatter(x = dataframe.world_rank,
                    y = dataframe.research,
                    name = 'research')
trace2 = go.Scatter(x = dataframe.world_rank,
                    y = dataframe.citations,
                    xaxis = 'x2',
                    yaxis = 'y2',
                    name = 'citations')
trace3 = go.Scatter(x = dataframe.world_rank,
                    y = dataframe.income,
                    xaxis = 'x3',
                    yaxis = 'y3',
                    name = 'income')
trace4 = go.Scatter(x = dataframe.world_rank,
                     y = dataframe.total_score,
                     xaxis = 'x4',
                     yaxis = 'y4',
                     name = 'total score')

# our data
data = [trace1, trace2, trace3, trace4]

# our layout
layout = go.Layout(xaxis = dict(domain = [0, 0.45]),
                   yaxis = dict(domain = [0, 0.45]),
                   xaxis2 = dict(domain = [0.55, 1], anchor = 'y2'),
                   xaxis3 = dict(domain = [0, 0.45], anchor = 'y3'),
                   xaxis4 = dict(domain = [0.55, 1], anchor = 'y4'),
                   yaxis2 = dict(domain = [0, 0.45], anchor = 'x2'),
                   yaxis3 = dict(domain = [0.55, 1], anchor = 'x3'),
                   yaxis4 = dict(domain = [0.55, 1], anchor = 'x4'),
                   title = 'research, citation, income and total score vs world rank of universities'
                  )
fig = go.Figure(data=data, layout=layout)
iplot(fig)
plt.savefig('multiple_subplots.png')
