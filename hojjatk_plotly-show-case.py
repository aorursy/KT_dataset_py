import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import numpy as np

import scipy as sp

import plotly 

plotly.tools.set_credentials_file(username='hojjatk', api_key='Bh8zInNKJMJkTFWDdWAi')

import plotly.plotly as py

import plotly.figure_factory as ff
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/school_earnings.csv")

table = ff.create_table(df)

py.iplot(table, filename='jupyter-table1')
import plotly.graph_objs as go



data = [go.Bar(x=df.School, y=df.Gap)]

py.iplot(data, filename='jupyter-basic_bar')
import plotly.plotly as py

import plotly.graph_objs as go



trace1 = go.Bar(

    x=[1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,

       2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],

    y=[219, 146, 112, 127, 124, 180, 236, 207, 236, 263,

       350, 430, 474, 526, 488, 537, 500, 439],

    name='Rest of world',

    marker=dict(

        color='rgb(55, 83, 109)'

    )

)

trace2 = go.Bar(

    x=[1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,

       2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],

    y=[16, 13, 10, 11, 28, 37, 43, 55, 56, 88, 105, 156, 270,

       299, 340, 403, 549, 499],

    name='China',

    marker=dict(

        color='rgb(26, 118, 255)'

    )

)

data = [trace1, trace2]

layout = go.Layout(

    title='US Export of Plastic Scrap',

    xaxis=dict(

        tickfont=dict(

            size=14,

            color='rgb(107, 107, 107)'

        )

    ),

    yaxis=dict(

        title='USD (millions)',

        titlefont=dict(

            size=16,

            color='rgb(107, 107, 107)'

        ),

        tickfont=dict(

            size=14,

            color='rgb(107, 107, 107)'

        )

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15,

    bargroupgap=0.1

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='style-bar')
import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)



df = pd.DataFrame({'a': np.random.randn(1000) + 1,

                   'b': np.random.randn(1000),

                   'c': np.random.randn(1000) - 1})



df.iplot(kind='histogram', filename='cufflinks/basic-histogram')
df = pd.DataFrame(np.random.randn(1000, 2), columns=['A', 'B']).cumsum()

df.iplot(filename='cufflinks/line-example')
import plotly.plotly as py

import plotly.graph_objs as go



labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']

values = [4500,2500,1053,500]



trace = go.Pie(labels=labels, values=values)



py.iplot([trace], filename='basic_pie_chart')
import plotly.plotly as py

import plotly.graph_objs as go



fig = {

  "data": [

    {

      "values": [16, 15, 12, 6, 5, 4, 42],

      "labels": [

        "US",

        "China",

        "European Union",

        "Russian Federation",

        "Brazil",

        "India",

        "Rest of World"

      ],

      "domain": {"column": 0},

      "name": "GHG Emissions",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },

    {

      "values": [27, 11, 25, 8, 1, 3, 25],

      "labels": [

        "US",

        "China",

        "European Union",

        "Russian Federation",

        "Brazil",

        "India",

        "Rest of World"

      ],

      "text":["CO2"],

      "textposition":"inside",

      "domain": {"column": 1},

      "name": "CO2 Emissions",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Global Emissions 1990-2011",

        "grid": {"rows": 1, "columns": 2},

        "annotations": [

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "GHG",

                "x": 0.20,

                "y": 0.5

            },

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "CO2",

                "x": 0.8,

                "y": 0.5

            }

        ]

    }

}

py.iplot(fig, filename='donut')
from IPython.display import IFrame

IFrame(src= "https://dash-simple-apps.plotly.host/dash-pieplot", width="100%", height="650px" ,frameBorder="0")
df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])

df.iplot(kind='box', filename='cufflinks/box-plots')
df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])

df.iplot(kind='area', fill=True, filename='cuflinks/stacked-area')

df.iplot(fill=True, filename='cuflinks/filled-area')
import pandas as pd

df = pd.read_csv('http://www.stat.ubc.ca/~jenny/notOcto/STAT545A/examples/gapminder/data/gapminderDataFiveYear.txt', sep='\t')

df2007 = df[df.year==2007]

df1952 = df[df.year==1952]



df2007.iplot(kind='scatter', mode='markers', x='gdpPercap', y='lifeExp', filename='cufflinks/simple-scatter')
cf.set_config_file(theme='white')

df2007.iplot(kind='bubble', x='gdpPercap', y='lifeExp', size='pop', text='country', xTitle='GDP per Capita', yTitle='Life Expectancy', filename='cufflinks/simple-bubble-chart')
cf.datagen.heatmap(20,20).iplot(kind='heatmap',colorscale='spectral', filename='cufflinks/simple-heatmap')