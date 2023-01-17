import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import seaborn as sns

# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

# ignore warnings

warnings.filterwarnings("ignore")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import os

print(os.listdir("../input"))
df = pd.read_csv("../input/Life Expectancy Data.csv")

print(plt.style.available) # look at available plot styles

plt.style.use('ggplot')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('-', '_').str.replace(',', '_')
import plotly.plotly as py

import plotly.graph_objs as go



trace1 = {"x": df.year, 

          "y": [ 80.709375, 81.1375, 80.68125, 80.44375, 80.70625, 80.146875, 79.584375, 78.93125, 79.3, 79.13125, 78.590625, 78.384375, 75.528125, 75.140625, 74.725, 74.403125 ], 

          "mode": "markers", 

          "name": "Developed", 

          "type": "scatter"

}



trace2 = {"x": df.year, 

          "y": [  69.69006623, 69.50198675, 69.23443709, 68.89801325, 68.52384106, 67.90860927, 67.89403974, 67.41390728, 66.86092715, 66.45033113, 66.00927152, 65.37086093, 65.20662252, 65.19072848, 65.00993377, 64.61986755  ], 

          "marker": {"color": "orange", "size": 12}, 

          "mode": "markers", 

          "name": "Developing", 

          "type": "scatter", 

}



data = [trace1, trace2]

layout = {"title": "Developed & Developing Countries Life Expectency Comparison", 

          "xaxis": {"title": "Year", }, 

          "yaxis": {"title": "Avarage Age"}}



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic_dot-plot')
dfded = df[df.status == "Developed"]

dfding = df[df.status == "Developing"]

import plotly.plotly as py

import plotly.graph_objs as go

import numpy as np



trace0 = go.Scatter(

    x = dfded.gdp,

    y = dfded.life_expectancy,

    name = 'Developed',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = 'rgba(255, 182, 193, .9)',

        line = dict(

            width = 2,

            color = 'rgb(0, 0, 0)'

        )

    )

)



trace1 = go.Scatter(

    x = dfding.gdp,

    y = dfding.life_expectancy,

    name = 'Developing',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = ' rgba(152, 0, 0, .8)',

        line = dict(

            width = 2,

        )

    )

)



data = [trace0, trace1]



layout = dict(title = 'GDP Per Capita and Life Expectancy Correlation',

              yaxis = dict(zeroline = False),

              xaxis = dict(zeroline = False)

             )



fig = dict(data=data, layout=layout)

iplot(fig, filename='styled-scatter')


booleandfed = dfded.schooling > 2

booleandfing= dfding.schooling > 2

dfded[booleandfed]

dfding[booleandfing]



trace0 = go.Scatter(

    x = dfded[booleandfed].schooling,

    y = dfded[booleandfed].life_expectancy,

    name = 'Developed',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = 'rgba(255, 182, 193, .9)',

        line = dict(

            width = 2,

            color = 'rgb(0, 0, 0)'

        )

    )

)



trace1 = go.Scatter(

    x = dfding[booleandfing].schooling,

    y = dfding[booleandfing].life_expectancy,

    name = 'Developing',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = ' rgba(152, 0, 0, .8)',

        line = dict(

            width = 2,

        )

    )

)



data = [trace0, trace1]



layout = dict(title = 'Education (Number of years of Schooling) and Life Expectancy Correlation',

              yaxis = dict(zeroline = False),

              xaxis = dict(zeroline = False)

             )



fig = dict(data=data, layout=layout)

iplot(fig, filename='styled-scatter')