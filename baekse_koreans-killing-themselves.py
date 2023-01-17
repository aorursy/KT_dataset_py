import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py



import plotly.graph_objs as go

import cufflinks as cf
py.offline.init_notebook_mode(connected=True)

pd.set_option('float_format', '{:.4f}'.format)



cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
data = pd.read_csv('../input/master.csv')
data.head()
south_korea = data[data.country == 'Republic of Korea']
year = south_korea.groupby('year').sum().index.tolist()
f = go.Scatter(

    x = year,

    y = south_korea[south_korea.sex == 'female'].groupby('year').sum()['suicides/100k pop'],

    mode = 'lines+markers',

    name = 'female'

)

m = go.Scatter(

    x = year,

    y = south_korea[south_korea.sex == 'male'].groupby('year').sum()['suicides/100k pop'],

    mode = 'lines+markers',

    name = 'male'

)

t = go.Scatter(

    x = year,

    y = south_korea.groupby('year').sum()['suicides/100k pop'],

    mode = 'lines+markers',

    name = 'total'

)

data = [f, m, t]



py.offline.iplot(data)