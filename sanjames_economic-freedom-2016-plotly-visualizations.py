#Import all required libraries for reading data, analysing and visualizing data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
import os
print(os.listdir("../input"))
#Read the data
efw = pd.read_csv('../input/efw_cc.csv')
efw.shape
efw.head(3)
efw.info()
efw.year.value_counts().sort_index().index
efw.ISO_code.unique()
efw2016 = efw[efw.year == 2016]
efw2016.shape
efw2016 = efw2016.fillna(0)
efw2016.isnull().sum()[efw2016.isnull().sum()>0]
efw2016.head()
efw2016_x = efw2016[['ISO_code', 'countries', 'ECONOMIC FREEDOM', 'rank', 'quartile','1_size_government', 
                     '2_property_rights', '3_sound_money', '4_trade', '5_regulation']]
top10_efw = efw2016_x.sort_values('ECONOMIC FREEDOM', ascending=False).head(11)
top10_efw.head()
import plotly.graph_objs as go
trace1 = go.Bar(
    x=top10_efw['countries'],
    y=top10_efw['1_size_government'],
    name='Size of Govt'
)
trace2 = go.Bar(
    x=top10_efw['countries'],
    y=top10_efw['2_property_rights'],
    name='Property Rights'
)
trace3 = go.Bar(
    x=top10_efw['countries'],
    y=top10_efw['3_sound_money'],
    name='Sound Money'
)
trace4 = go.Bar(
    x=top10_efw['countries'],
    y=top10_efw['4_trade'],
    name='Freedom to Trade'
)
trace5 = go.Bar(
    x=top10_efw['countries'],
    y=top10_efw['5_regulation'],
    name='Regulation'
)

data = [trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(
    title='Top 10 countries & Economic indicators',
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='Top 10 countries and different indicators')

import plotly.graph_objs as go
trace0 = go.Bar(
    x=top10_efw['ECONOMIC FREEDOM'],
    y=top10_efw['ISO_code'],
    marker=dict(
        color='rgba(66, 244, 146, 0.6)',
        line=dict(
            color='rgba(66, 244, 146, 1.0)',
            width=1),
    ),
    name='ECONOMIC FREEDOM',
    orientation='h',    
)
trace1 = go.Scatter(
    x=top10_efw['1_size_government'],
    y=top10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(214, 53, 25)'),
    name='Government',
)
trace2 = go.Scatter(
    x=top10_efw['2_property_rights'],
    y=top10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(192, 26, 221)'),
    name='Property Rights',
)
trace3 = go.Scatter(
    x=top10_efw['3_sound_money'],
    y=top10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(114, 78, 22)'),
    name='Sound Money',
)
trace4 = go.Scatter(
    x=top10_efw['4_trade'],
    y=top10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(14, 94, 160)'),
    name='Trade',
)
trace5 = go.Scatter(
    x=top10_efw['5_regulation'],
    y=top10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(237, 135, 147)'),
    name='Regulation',
)
layout = dict(
    title='World Economic Freedom 2016 - Top10 countries',
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
        domain=[0, 0.32],
    ),
    yaxis2=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0, 0.32],
    ),
    yaxis3=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.34, .66],
    ),
    yaxis4=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.34, .66],
    ),    
    yaxis5=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.68, 1],
    ),
    yaxis6=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.68, 1],
    ),        
    xaxis=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0, 0.45],
    ),
    xaxis2=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0.55, 1],
        side='top',
        dtick=25000,
    ),
    xaxis3=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0, 0.45],
    ),
    xaxis4=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0.55, 1],
        side='top',
        dtick=25000,
    ),  
    xaxis5=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0, 0.40],
        side='top',
        dtick=25000,
    ),
    xaxis6=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0.55, 1],
        side='top',
        dtick=25000,
    ),    

    legend=dict(
        x=0.029,
        y=1.038,
        font=dict(
            size=10,
        ),
    ),
    margin=dict(
        l=100,
        r=20,
        t=70,
        b=70,
    ),    
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)


# Creating two subplots
fig = tools.make_subplots(rows=3, cols=2, specs=[[{}, {}],[{}, {}],[{}, {}]], shared_xaxes=False, shared_yaxes=False,
                         subplot_titles=('Economic Freedom', 'Govt spending, decision making','Legal System & Property Rights',
                                         'Sound Money', 'Freedom to Trade', 'Regulation'), vertical_spacing=0.1)

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 2, 2)
fig.append_trace(trace4, 3, 1)
fig.append_trace(trace5, 3, 2)


fig['layout'].update(height=1000, width=1000,  title='World Economic Freedom 2016 - Top10 countries')
iplot(fig, filename='ECONOMIC FREEDOM Vs Govt')
low10_efw = efw2016_x.sort_values('ECONOMIC FREEDOM', ascending=False).tail(10)
low10_efw.head()
import plotly.graph_objs as go

trace1 = go.Bar(
    x=low10_efw['countries'],
    y=low10_efw['1_size_government'],
    name='Size of Govt'
)
trace2 = go.Bar(
    x=low10_efw['countries'],
    y=low10_efw['2_property_rights'],
    name='Property Rights'
)
trace3 = go.Bar(
    x=low10_efw['countries'],
    y=low10_efw['3_sound_money'],
    name='Sound Money'
)
trace4 = go.Bar(
    x=low10_efw['countries'],
    y=low10_efw['4_trade'],
    name='Freedom to Trade'
)
trace5 = go.Bar(
    x=low10_efw['countries'],
    y=low10_efw['5_regulation'],
    name='Regulation'
)

data = [trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(
    title='Lowest 10 countries & Economic indicators',    
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='Lowest rated countries and different indicators')
import plotly.graph_objs as go

trace0 = go.Bar(
    x=low10_efw['ECONOMIC FREEDOM'],
    y=low10_efw['ISO_code'],
    marker=dict(
        color='rgba(66, 244, 146, 0.6)',
        line=dict(
            color='rgba(66, 244, 146, 1.0)',
            width=1),
    ),
    name='ECONOMIC FREEDOM',
    orientation='h',    
)
trace1 = go.Scatter(
    x=low10_efw['1_size_government'],
    y=low10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(214, 53, 25)'),
    name='Government',
)
trace2 = go.Scatter(
    x=low10_efw['2_property_rights'],
    y=low10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(192, 26, 221)'),
    name='Property Rights',
)
trace3 = go.Scatter(
    x=low10_efw['3_sound_money'],
    y=low10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(114, 78, 22)'),
    name='Sound Money',
)
trace4 = go.Scatter(
    x=low10_efw['4_trade'],
    y=low10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(14, 94, 160)'),
    name='Trade',
)
trace5 = go.Scatter(
    x=low10_efw['5_regulation'],
    y=low10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(237, 135, 147)'),
    name='Regulation',
)
layout = dict(
    title='World Economic Freedom 2016 - Lowest rated countries',
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
        domain=[0, 0.32],
    ),
    yaxis2=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0, 0.32],
    ),
    yaxis3=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.34, .66],
    ),
    yaxis4=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.34, .66],
    ),    
    yaxis5=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.68, 1],
    ),
    yaxis6=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.68, 1],
    ),        
    xaxis=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0, 0.45],
    ),
    xaxis2=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0.55, 1],
        side='top',
        dtick=25000,
    ),
    xaxis3=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0, 0.45],
    ),
    xaxis4=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0.55, 1],
        side='top',
        dtick=25000,
    ),  
    xaxis5=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0, 0.40],
        side='top',
        dtick=25000,
    ),
    xaxis6=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0.55, 1],
        side='top',
        dtick=25000,
    ),    

    legend=dict(
        x=0.029,
        y=1.038,
        font=dict(
            size=10,
        ),
    ),
    margin=dict(
        l=100,
        r=20,
        t=70,
        b=70,
    ),    
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)


# Creating two subplots
fig = tools.make_subplots(rows=3, cols=2, specs=[[{}, {}],[{}, {}],[{}, {}]], shared_xaxes=False, shared_yaxes=False,
                         subplot_titles=('Economic Freedom', 'Govt spending, decision making','Legal System & Property Rights',
                                         'Sound Money', 'Freedom to Trade', 'Regulation'), vertical_spacing=0.1)

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 2, 2)
fig.append_trace(trace4, 3, 1)
fig.append_trace(trace5, 3, 2)


fig['layout'].update(height=1000, width=1000,  title='World Economic Freedom 2016 - Lowest rated countries')
iplot(fig, filename='ECONOMIC FREEDOM Vs Govt')
import plotly.graph_objs as go
data = [ dict(
        type = 'choropleth',
        locations = efw2016['ISO_code'],
        z = efw2016['quartile'],
        text = efw2016['countries'],
        colorscale = 'Rainbow',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'World Economic Index'),
      ) ]

layout = dict(
    title = '2016 World Economic Freedom Index',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='Free Economies' )

import plotly.graph_objs as go
data = [ dict(
        type = 'choropleth',
        locations = efw2016['ISO_code'],
        z = efw2016['1_size_government'],
        text = efw2016['countries'],
        colorscale = 'Rainbow',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'World Economic Index'),
      ) ]

layout = dict(
    title = '2016 World Economic Freedom - Government Controlled Index',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='Free Economies' )

import plotly.graph_objs as go
data = [ dict(
        type = 'choropleth',
        locations = efw2016['ISO_code'],
        z = efw2016['2_property_rights'],
        text = efw2016['countries'],
        colorscale = 'Hot',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Money'),
      ) ]

layout = dict(
    title = '2016 World Economic Freedom - Property Rights & Legal',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='Free Economies' )

import plotly.graph_objs as go
data = [ dict(
        type = 'choropleth',
        locations = efw2016['ISO_code'],
        z = efw2016['3_sound_money'],
        text = efw2016['countries'],
        colorscale = 'Hot',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Money'),
      ) ]

layout = dict(
    title = '2016 World Economic Freedom - Sound Money',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='Free Economies' )

import plotly.graph_objs as go
data = [ dict(
        type = 'choropleth',
        locations = efw2016['ISO_code'],
        z = efw2016['4_trade'],
        text = efw2016['countries'],
        colorscale = 'Hot',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'International Trade'),
      ) ]

layout = dict(
    title = '2016 World Economy - Freedom to trade internationally',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='Free Economies' )

import plotly.graph_objs as go
data = [ dict(
        type = 'choropleth',
        locations = efw2016['ISO_code'],
        z = efw2016['5_regulation'],
        text = efw2016['countries'],
        colorscale = 'Hot',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'REgulations'),
      ) ]

layout = dict(
    title = '2016 World Economy - Regulations',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='Free Economies' )

