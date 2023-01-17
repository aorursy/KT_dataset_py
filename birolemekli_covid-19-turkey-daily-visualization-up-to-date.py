import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px
df=pd.read_csv("/kaggle/input/covid19turkeydailydetailsdataset/covid19-Turkey.csv",index_col="date",parse_dates=True)

df.head(5)
fig = go.Figure()

fig.add_trace(go.Bar(x=df.index,

                y=df.totalCases,

                name='Cases',

                marker_color='rgb(0, 0, 255)'

                ))

fig.add_trace(go.Bar(x=df.index,

                y=df.totalRecovered,

                name='Recovered',

                marker_color='rgb(0,255, 0)'

                ))

fig.add_trace(go.Bar(x=df.index,

                y=df.totalDeaths,

                name='Deaths',

                marker_color='rgb(255, 0, 0)'

                ))

fig.update_layout(

    title='Total Cases Time Series',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Sayı',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.03,

    bargroupgap=0.1

)

fig.show()
labels=['Cases','Recovered','Deaths']

values=[df.totalCases.max()-df.totalRecovered.max(),df.totalRecovered.max(),df.totalDeaths.max()]

irises_colors = ['rgb(0, 0, 255)', 'rgb(50, 200, 110)', 'rgb(255, 0, 0)']

fig = make_subplots(1, specs=[[{'type':'domain'}]],subplot_titles=['Türkiye'])

fig.add_trace(go.Pie(labels=labels, values=values, pull=[0,0.1,0.12], hole=.4,marker_colors=irises_colors))

fig.update_layout(title_text='Turkey Total Active Cases')

fig.show()
labels=['Cases','Intensive Care','Intubated']

values=[df.totalCases.max()-df.totalRecovered.max(),df.totalIntensiveCare.max(),df.totalIntubated.max()]

irises_colors = ['rgb(0, 0, 255)', 'rgb(175, 0, 110)', 'rgb(255, 0, 0)']

fig = make_subplots(1, specs=[[{'type':'domain'}]],subplot_titles=['Türkiye'])

fig.add_trace(go.Pie(labels=labels, values=values, pull=[0,0.1,0.12], hole=.4,marker_colors=irises_colors))

fig.update_layout(title_text='Total Case, Intensive Care and Intubated Patient Rates')

fig.show()
fig = px.line(df, x=df.index, y=[df.dailyTests,df.dailyCases])

fig.update_layout(

    title='Daily Test Cases Time Series',

    yaxis=dict(

        title='Count'

    )

)

fig.show()
title="daily case status time series"

fig = go.Figure()

fig.add_trace(go.Scatter(x=df.index, y=df.totalCases,

                    marker_color='blue',

                    mode='markers',

                    name='Cases'))

fig.add_trace(go.Scatter(x=df.index, y=df.totalRecovered,

                    marker_color='green',

                    mode='lines+markers',

                    name='Recovered'))

fig.add_trace(go.Scatter(x=df.index, y=df.totalDeaths,

                    marker_color='red',

                    mode='lines',

                    name='Deaths'))

fig.update_layout(

    title=title.title(),

    xaxis_title="Date",

    yaxis_title="Count"

)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=df.index, y=df.dailyCases,

                    marker_color='blue',

                    mode='lines',

                    name='Daily Cases'))

fig.add_trace(go.Scatter(x=df.index, y=df.dailyRecovered,

                    marker_color='green',

                    mode='lines+markers',

                    name='Daily Recovered'))

fig.update_layout(title='Daily Cases Recovered Time Series',

                  yaxis_zeroline=False, xaxis_zeroline=False)

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(x=df.index,

                y=df.dailyCases,

                name='Daily Cases',

                marker_color='rgb(0, 0, 255)'

                ))

fig.add_trace(go.Bar(x=df.index,

                y=df.dailyRecovered,

                name='Daily Recovered',

                marker_color='rgb(0,255, 0)'

                ))

fig.add_trace(go.Bar(x=df.index,

                y=df.dailyDeaths,

                name='Daily Deaths',

                marker_color='rgb(255, 0, 0)'

                ))

fig.update_layout(

    title='Daily Status',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Count',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.03,

    bargroupgap=0.1

)

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Daily Test', x=df.index, y=df.dailyTests),

    go.Bar(name='Daily Cases', x=df.index, y=df.dailyCases)

])

fig.update_layout(

    title='Dailt Test Cases',

    yaxis=dict(

        title='Count'

    )

)

fig.update_layout(barmode='stack')
fig = px.line(df, x=df.index, y=[df.dailyCases,df.totalIntensiveCare,df.totalIntubated])

fig.update_layout(

    title='Those with severe disease',

    yaxis=dict(

        title='Count'

    )

)

fig.show()
fig = px.line(df, x=df.index, y=[df.totalDeaths,df.totalIntensiveCare,df.totalIntubated])

fig.update_layout(

    title='People with severe disease who died',

    yaxis=dict(

        title='Count'

    )

)

fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=df.index, y=df.totalIntensiveCare,

    name='Total Intensive Care',

    mode='markers',

    marker_color='rgba(152, 0, 0, .8)'

))



fig.add_trace(go.Scatter(

    x=df.index, y=df.totalIntubated,

    name='Total Intubated',

    marker_color='rgba(255, 182, 193, .9)'

))

fig.add_trace(go.Scatter(

    x=df.index, y=df.dailyCases,

    name='Daily Cases',

    marker_color='rgba(0, 5, 200, .9)'

))

fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)

fig.update_layout(title='Intensive Care and Intubated Time Series',

                  yaxis_zeroline=False, xaxis_zeroline=False)

fig.show()