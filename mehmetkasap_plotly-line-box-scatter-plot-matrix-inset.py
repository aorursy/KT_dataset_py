# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
timesData.head()
df = timesData.iloc[:100, :]
trace1 = go.Scatter (x = df.world_rank,
                     y = df.citations,
                     mode = 'lines',
                     name = 'citations',
                     marker = dict(color = 'rgba(16, 122, 2, 0.8)'),
                     text = df.university_name)
trace2 = go.Scatter (x = df.world_rank,
                     y = df.teaching,
                     mode = 'lines+markers',
                     name = 'teaching',
                     marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                     text = df.university_name)
data = [trace1, trace2]
layout = dict(title = 'Citation and Teaching vs World Rank of top 100 Universities',
              xaxis = dict(title = 'World Rank', ticklen = 5, zeroline = False)
              )

fig = dict(data = data, layout = layout)
iplot(fig)
plt.savefig('line_plot_using_plotly.png')
plt.show()
timesData.columns
# data preparation
x2015 = timesData[timesData.year == 2015]

trace0 = go.Box(
    y=x2015.total_score,
    name = 'total score of universities in 2015',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=x2015.research,
    name = 'research of universities in 2015',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)
plt.savefig('box-plot.png')
dataframe = timesData[timesData.year == 2015]
data2015 = dataframe.loc[:,["research","international", "total_score"]]
# import figure factory
import plotly.figure_factory as ff

# data
dataframe = timesData[timesData.year == 2015]
data2015 = dataframe.loc[:,["research","international", "total_score"]]

# add index feauture in data2015
data2015 ['index'] = np.arange(1,len(data2015)+1)

# figure
# in parantesysis we have (our data, plot we want to see in the diagonal, index, color map,..)
fig = ff.create_scatterplotmatrix (data2015, diag='box', index='index', colormap='Portland',
                                  colormap_type='cat', height=700, width=700)
iplot(fig)
plt.savefig('scatterplotmatrix.png')
plt.show()
dataframe.columns
# first plot
trace1 = go.Scatter (x = dataframe.world_rank,
                     y = dataframe.teaching,
                     name = 'teaching',
                     marker = dict(color = 'rgba(16,112,2,0.8)'),
                    )
# second line
trace2 = go.Scatter (x = dataframe.world_rank,
                     y = dataframe.income,
                     xaxis = 'x2',
                     yaxis = 'y2',
                     name = 'income',
                     marker = dict(color = 'rgba(160,112,20,0.8)'))

data = [trace1, trace2]
layout = go.Layout (xaxis2 = dict(domain = [0.6, 0.95], anchor = 'y2'),
                    yaxis2 = dict(domain = [0.6, 0.95], anchor = 'x2'),
                    title = 'income and teaching vs world rank of universities')

fig = go.Figure(data=data, layout=layout)
iplot(fig)
plt.savefig('inset-plot.png')
