
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
import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import datetime as dt
from fbprophet import Prophet

import warnings
warnings.filterwarnings("ignore")

# plt.style.available
plt.style.use("seaborn-whitegrid")

df = pd.read_csv('../input/all_stocks_5yr.csv')
df.head()
df.tail()
df.describe()
df.info()
df['date'] = pd.to_datetime(df['date'])
df.info()
df.Name.unique()
nflx = df.loc[df['Name']== 'NFLX']
nflx.head()
c = df.loc[df['Name']== 'C']
c.head()
f, (ax1, ax2) = plt.subplots( 1,2 , figsize = (20,12))
ax1.plot( nflx['date'], nflx['close'],color = 'blue')
ax1.plot(c['date'], c['close'], color = 'orange')
ax1.set_xlabel('Year')
ax1.set_ylabel('Stock Price')


ax2.plot(nflx['date'], nflx['volume'], color= 'blue')
ax2.plot(c['date'], c['volume'], color = 'orange')
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
from plotly import tools
import plotly.figure_factory as ff


trace0 = go.Scatter(x=nflx.date, y=nflx.close)
trace1 = go.Scatter(x= c.date, y = c.close)
data = [trace0, trace1]
py.iplot(data)

trace1 = go.Scatter(x=nflx['date'], y=nflx['close'])
trace2 = go.Scatter(x=c['date'], y=c['close'])

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Netflix', 'Citi Bank'))

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)

fig['layout'].update(height=800, width=1000, title='Multiple Subplots' +
                                                  ' with Titles')


py.iplot(fig, filename='make-subplots-multiple-with-titles')

trace0 = go.Box(
    y=nflx.close,
    name = 'Netflix Close',
    marker = dict(
        color = 'red')
)

trace1 = go.Box(
    y=c.close,
    name = ' Citi Bank Close',
    marker = dict(
        color = 'navy')
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Netflix', 'Citi Bank'))

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=400, width=600, title='Stock Prices of Netflix & Citi Bank')


py.iplot(fig, filename='Stock Prices of Netflix & Citi Bank')
trace0 = go.Candlestick(x=nflx.date,
                        open=nflx.open,
                        high=nflx.high, 
                        low=nflx.low,
                        close=nflx.close
                       )
data = [trace0]
py.iplot(data)
trace1 = go.Candlestick(x=c.date,
                        open=c.open,
                        high=c.high, 
                        low=c.low,
                        close=c.close
                       )
data = [trace1]
py.iplot(data)