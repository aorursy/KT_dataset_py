# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# import graph objects as "go"

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os
gdp=pd.read_csv('../input/worldwide-gdp-history-19602016/gdp_data.csv',encoding='ISO-8859-1')
argentinaData = gdp[gdp['Country'] == 'Argentina']

chileData = gdp[gdp['Country'] == 'Chile']

brazilData = gdp[gdp['Country'] == 'Brazil']

trace1 = go.Scatter(

                x = argentinaData['Year'],

                y = argentinaData['GDP'],

                name = "Argentina",

                marker=dict(

                        size=9,

                        color = ('aqua'))

)

trace2 = go.Scatter(

                x = chileData['Year'],

                y = chileData['GDP'],

                name = "Chile",

                marker=dict(

                    size=9,

                    color = ('navy'))

    )

trace3 = go.Scatter(

                x = brazilData['Year'],

                y = brazilData['GDP'],

                name = "Brazil",

                marker=dict(

                    size=9,

                    color = ('red'))

    )

data = [trace1, trace2, trace3]

layout = dict(title = 'GDP ($)',

              xaxis = dict(title = 'Year'),

              yaxis = dict(title = 'GDP ($)')

             )

fig = go.Figure(data = data, layout = layout)

iplot(fig)
argentinaData = gdp[gdp['Country'] == 'Argentina']

chileData = gdp[gdp['Country'] == 'Chile']

brazilData = gdp[gdp['Country'] == 'Brazil']

trace1 = go.Scatter(

                x = argentinaData['Year'],

                y = argentinaData['GDP-Per-Capita'],

                name = "Argentina",

                marker=dict(

                        size=9,

                        color = ('aqua'))

)

trace2 = go.Scatter(

                x = chileData['Year'],

                y = chileData['GDP-Per-Capita'],

                name = "Chile",

                marker=dict(

                    size=9,

                    color = ('navy'))

    )

trace3 = go.Scatter(

                x = brazilData['Year'],

                y = brazilData['GDP-Per-Capita'],

                name = "Brazil",

                marker=dict(

                    size=9,

                    color = ('red'))

    )

data = [trace1, trace2, trace3]

layout = dict(title = 'GDP Per Capita ($)',

              xaxis = dict(title = 'Year'),

              yaxis = dict(title = 'GDP Per Capita ($)')

             )

fig = go.Figure(data = data, layout = layout)

iplot(fig)
argentinaData = gdp[gdp['Country'] == 'Argentina']

chileData = gdp[gdp['Country'] == 'Chile']

brazilData = gdp[gdp['Country'] == 'Brazil']

trace1 = go.Scatter(

                x = argentinaData['Year'],

                y = argentinaData['GDP-Growth'],

                name = "Argentina",

                marker=dict(

                        size=9,

                        color = ('aqua'))

)

trace2 = go.Scatter(

                x = chileData['Year'],

                y = chileData['GDP-Growth'],

                name = "Chile",

                marker=dict(

                    size=9,

                    color = ('navy'))

    )

trace3 = go.Scatter(

                x = brazilData['Year'],

                y = brazilData['GDP-Growth'],

                name = "Brazil",

                marker=dict(

                    size=9,

                    color = ('red'))

    )

data = [trace1, trace2, trace3]

layout = dict(title = 'GDP-Growth ($)',

              xaxis = dict(title = 'Year'),

              yaxis = dict(title = 'GDP-Growth ($)')

             )

fig = go.Figure(data = data, layout = layout)

iplot(fig)
import pandas as pd

GLOB_SES = pd.read_csv("../input/globses/GLOB.SES.csv", encoding='iso-8859-1')
ArgCoef = GLOB_SES[GLOB_SES['country']=='Argentina'].sort_values(by=['year'])

BrazCoef = GLOB_SES[GLOB_SES['country']=='Brazil'].sort_values(by=['year'])

ChileCoef = GLOB_SES[GLOB_SES['country']=='Chile'].sort_values(by=['year'])



trace1 = go.Scatter(

                x = ArgCoef['year'],

                y = ArgCoef['popshare'],

                name = "Argentina",

                marker=dict(

                        size=9,

                        color = ('aqua'))

)

trace2 = go.Scatter(

                x = ChileCoef['year'],

                y = ChileCoef['popshare'],

                name = "Chile",

                marker=dict(

                    size=9,

                    color = ('navy'))

    )

trace3 = go.Scatter(

                x = BrazCoef['year'],

                y = BrazCoef['popshare'],

                name = "Brazil",

                marker=dict(

                    size=9,

                    color = ('red'))

    )

data = [trace1, trace2, trace3]

layout = dict(title = 'Sharing population',

              xaxis = dict(title = 'Year'),

              yaxis = dict(title = 'PopShare')

             )

fig = go.Figure(data = data, layout = layout)

iplot(fig)
ArgCoef = GLOB_SES[GLOB_SES['country']=='Argentina'].sort_values(by=['year'])

BrazCoef = GLOB_SES[GLOB_SES['country']=='Brazil'].sort_values(by=['year'])

ChileCoef = GLOB_SES[GLOB_SES['country']=='Chile'].sort_values(by=['year'])



trace1 = go.Scatter(

                x = ArgCoef['year'],

                y = ArgCoef['SES'],

                name = "Argentina",

                marker=dict(

                        size=9,

                        color = ('aqua'))

)

trace2 = go.Scatter(

                x = ChileCoef['year'],

                y = ChileCoef['SES'],

                name = "Chile",

                marker=dict(

                    size=9,

                    color = ('navy'))

    )

trace3 = go.Scatter(

                x = BrazCoef['year'],

                y = BrazCoef['SES'],

                name = "Brazil",

                marker=dict(

                    size=9,

                    color = ('red'))

    )

data = [trace1, trace2, trace3]

layout = dict(title = 'SES',

              xaxis = dict(title = 'Year'),

              yaxis = dict(title = 'SES')

             )

fig = go.Figure(data = data, layout = layout)

iplot(fig)
ArgCoef = GLOB_SES[GLOB_SES['country']=='Argentina'].sort_values(by=['year'])

BrazCoef = GLOB_SES[GLOB_SES['country']=='Brazil'].sort_values(by=['year'])

ChileCoef = GLOB_SES[GLOB_SES['country']=='Chile'].sort_values(by=['year'])



trace1 = go.Scatter(

                x = ArgCoef['year'],

                y = ArgCoef['yrseduc'],

                name = "Argentina",

                marker=dict(

                        size=9,

                        color = ('aqua'))

)

trace2 = go.Scatter(

                x = ChileCoef['year'],

                y = ChileCoef['yrseduc'],

                name = "Chile",

                marker=dict(

                    size=9,

                    color = ('navy'))

    )

trace3 = go.Scatter(

                x = BrazCoef['year'],

                y = BrazCoef['yrseduc'],

                name = "Brazil",

                marker=dict(

                    size=9,

                    color = ('red'))

    )

data = [trace1, trace2, trace3]

layout = dict(title = 'yrseduc',

              xaxis = dict(title = 'Year'),

              yaxis = dict(title = 'yrseduc')

             )

fig = go.Figure(data = data, layout = layout)

iplot(fig)
import pandas as pd

Continent_HDI = pd.read_csv("../input/human-development-index-hdi/Continent_HDI.csv")

HDI = pd.read_csv("../input/human-development-index-hdi/HDI.csv")