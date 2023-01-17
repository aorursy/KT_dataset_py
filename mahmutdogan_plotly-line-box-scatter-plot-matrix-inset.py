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
timesData = pd.read_csv('/kaggle/input/arabam3/arabam3.csv')

timesData.info()
timesData.head()
df = timesData.iloc[:100, :]
trace1 = go.Scatter (x = df.zaman,

                     y = df.hız,

                     mode = 'lines',

                     name = 'hız',

                     marker = dict(color = 'rgba(16, 122, 2, 0.8)')

                    )
data = [trace1]

layout = dict(title = 'Hız - Zaman',

              xaxis = dict(title = 'Zaman', ticklen = 5, zeroline = False)

              )



fig = dict(data = data, layout = layout)

iplot(fig)

plt.savefig('line_plot_using_plotly.png')

plt.show()