import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns
levels = pd.read_csv("/kaggle/input/chennai-water-management/chennai_reservoir_levels.csv")

rainfall = pd.read_csv("/kaggle/input/chennai-water-management/chennai_reservoir_rainfall.csv")
levels.head()
rainfall.head()
levels.tail()
rainfall.tail()
import plotly.offline as py

from plotly import tools

import plotly.graph_objs as go
levels.Date = pd.to_datetime(levels.Date, format="%d-%m-%Y")



def scatter_plot(cnt_srs, colour):

    trace = go.Scatter(

       x = cnt_srs.index[::], 

       y = cnt_srs.values[::], 

       showlegend = False,

       mode = 'markers',

       marker = dict(color=colour,),

    )

    return trace





cnt_srs = levels.POONDI

cnt_srs.index = levels.Date

trace1 = scatter_plot(cnt_srs, 'blue')



cnt_srs = levels.CHOLAVARAM

cnt_srs.index = levels.Date

trace2 = scatter_plot(cnt_srs, 'red')



cnt_srs = levels.REDHILLS

cnt_srs.index = levels.Date

trace3 = scatter_plot(cnt_srs, 'black')



cnt_srs = levels.CHEMBARAMBAKKAM

cnt_srs.index = levels.Date

trace4 = scatter_plot(cnt_srs, 'green')



subtitles = ["Water Availability in Poondi reservoir - in mcft",

             "Water Availability in Cholavaram reservoir - in mcft",

             "Water Availability in Redhills reservoir - in mcft",

             "Water Availability in Chembarambakkam reservoir - in mcft"

            ]

fig = tools.make_subplots(rows=4, cols=1, vertical_spacing=0.08, subplot_titles=subtitles)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 3, 1)

fig.append_trace(trace4, 4, 1)



fig['layout'].update(height=1200, width=800,) #paper_bgcolor='rgb(233,233,233)')

py.iplot(fig)
#RainFall



rainfall.Date = pd.to_datetime(rainfall.Date, format="%d-%m-%Y")

rainfall['Total'] = rainfall.POONDI + rainfall.CHOLAVARAM + rainfall.REDHILLS + rainfall.CHEMBARAMBAKKAM



def bar_plot(cnt, color):

    trace = go.Bar(

        x = cnt.index[::],

        y = cnt.values[::],

        showlegend=False,

        marker = dict(color=color,line=dict(width=0.3, color='rgb(0,0,0)'),), 

    )

    return trace



rain_srs = rainfall.POONDI

rain_srs.index = rainfall.Date

trace5 = bar_plot(rain_srs, 'blue')



rain_srs = rainfall.CHOLAVARAM

rain_srs.index = rainfall.Date

trace6 = bar_plot(rain_srs, 'red')



rain_srs = rainfall.REDHILLS

rain_srs.index = rainfall.Date

trace7 = bar_plot(rain_srs, 'black')



rain_srs = rainfall.CHEMBARAMBAKKAM

rain_srs.index = rainfall.Date

trace8 = bar_plot(rain_srs, 'green')



title = ["Water Rainfall in Poondi reservoir - in mcft",

         "Water Rainfall in Cholavaram reservoir - in mcft",

         "Water Rainfall in Redhills reservoir - in mcft",

         "Water Rainfall in Chembarambakkam reservoir - in mcft" ]



fig = tools.make_subplots(rows=4, cols=1, shared_xaxes=False, vertical_spacing=0.09, subplot_titles=title)

fig['layout'].update(height=900, width=1000)



fig.append_trace(trace5, 1, 1)

fig.append_trace(trace6, 2, 1)

fig.append_trace(trace7, 3, 1)

fig.append_trace(trace8, 4, 1)



py.iplot(fig)
#Total RainFall Water



#rainfall.Date = pd.to_datetime(rainfall.Date, format="%d-%m-%Y")

rainfall.set_index('Date', inplace=True, drop=False)

#rainfall.head()

df = rainfall.groupby(pd.Grouper(freq='M')).sum()

df1 = df.Total

df1.head()

trace9 = bar_plot(df1, "pink")



fig = tools.make_subplots(rows=1, cols=1, shared_xaxes=False, subplot_titles=["Total Rainfall in Reservior-mm"])

fig['layout'].update(height=900, width=1000)



fig.append_trace(trace9, 1, 1)

py.iplot(fig)