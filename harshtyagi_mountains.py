import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import time

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

plt.rcParams['figure.figsize'] = 8,6
dataset = pd.read_csv('../input/Mountains.csv')
dataset.describe()
filter1 = dataset['Height (m)'] == 7200

filter2 = dataset['Height (m)'] == 8848

filter3 = dataset['Failed attempts bef. 2004'] == 121

filter4 = dataset['Failed attempts bef. 2004'] == 0
### Minimum Height



dataset[filter1]
dataset[filter2]
dataset[filter3]
dataset[filter4]
data = [go.Bar(

            x = dataset['Range'] ,

            y = dataset['Height (m)']

    )]



py.iplot(data, filename='basic-bar')













data = [go.Bar(

            x = dataset['Parent mountain'] ,

            y = dataset['Height (m)']

    )]



py.iplot(data, filename='basic-bar')



data = [go.Bar(

            x = dataset['Mountain'] ,

            y = dataset['Failed attempts bef. 2004']

    )]



py.iplot(data, filename='basic-bar')





trace = go.Scatter(

    x = dataset['First ascent'],

    y = dataset['Mountain'],

    mode = 'markers'

)



data = [trace]

py.iplot(data, filename='basic-scatter')


