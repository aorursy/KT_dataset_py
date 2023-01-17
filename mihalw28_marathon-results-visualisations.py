# Imports
import matplotlib.pyplot as plt 
import matplotlib
import plotly
import plotly.offline as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly.tools import make_subplots
import plotly.figure_factory as ff
init_notebook_mode()
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import pendulum
import datetime
# Read data
results = pd.read_csv('../input/marathon_results.csv')
results[:5]
# Add new column with finish time in seconds
results['Finish_time[m]'] = 0
start = pendulum.datetime(2018, 10, 14, 9, 0, 0)

def finish_time_in_min(df):
    for i in range(0, len(df)):
        end = pendulum.parse(df.iloc[i,15])
        delta = end - start
        df.iloc[i,18] = delta.total_minutes()
    return df

results = finish_time_in_min(results)
#Check categories
print(results.Cat.unique())
# Histogram
x1 = results['Finish_time[m]'][results['Sex'] == 'K']
x0 = results['Finish_time[m]'][results['Sex'] != 'K']

trace1 = go.Histogram(
    x=x0,
    name='M',
    xbins=dict(
        start=130,
        end=420,
        size= 'M33'),
    autobinx = False,
    opacity=0.6,
    marker=dict(
        color='#90BCC6'
    ),
)
trace2 = go.Histogram(
    x=x1,
    name='F',
    xbins=dict(
        start=130,
        end=420,
        size= 'M33'),
    autobinx = False,
    opacity=0.6,
    marker=dict(
        color='#9a90c6'
    )
)

layout=go.Layout(
    title='Finish line time vs. # of runners.',
    titlefont=dict(
        size=20,
        family='Droid Serif',
        color='#3B5F6D',
    ),
    barmode='overlay',
    yaxis=dict(
        title='No. of runners',
        titlefont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D'
        ),
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='rgb(243, 243, 243)'
    ),
    xaxis=dict(
        title='Finish time [minutes]',
        titlefont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        tickmode='array',
        tickvals=[150, 180, 210, 240, 270, 300, 330, 360, 390],
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='rgb(243, 243, 243)',
    ),
    legend=dict(
        orientation='h',
        font=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D'
        ),
        x=0.8,
        y=0.94,
    ),
    shapes=[
        dict(
            layer='below',
            type='line',
            x0=180,
            y0=70,
            x1=180,
            y1=0,
            line=dict(
                color='#335b66',
                width=1,
            ),
        ),
        dict(
            layer='below',
            type='line',
            x0=210,
            y0=70,
            x1=210,
            y1=0,
            line=dict(
                color='#335b66',
                width=1,
            ),
        ),
        dict(
            layer='below',
            type='line',
            x0=240,
            y0=70,
            x1=240,
            y1=0,
            line=dict(
                color='#335b66',
                width=1,
            ),
        ),
        dict(
            layer='below',
            type='line',
            x0=255,
            y0=70,
            x1=255,
            y1=0,
            line=dict(
                color='#335b66',
                width=1,
            ),
        ),
    ],
    annotations=[
        dict(
            x=173,
            y=68,
            xref='x',
            yref='y',
            text='3:00h',
            showarrow=False,
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
        dict(
            x=203,
            y=68,
            xref='x',
            yref='y',
            text='3:30h',
            showarrow=False,
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
        dict(
            x=233,
            y=68,
            xref='x',
            yref='y',
            text='4:00h',
            showarrow=False,
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
        dict(
            x=262,
            y=68,
            xref='x',
            yref='y',
            text='4:15h',
            showarrow=False,
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
        
    ],
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
# Find median for men & women
f_median = results['Finish_time[m]'][results['Sex'] == 'K'].median()
m_median = results['Finish_time[m]'][results['Sex'] != 'K'].median()
print(f'Median finish time for women is {round(f_median, 2)} minutes and for men is {round(m_median, 2)} minutes.')
# Stacked histogram - women
female18 = results['Finish_time[m]'][results['Cat'] == 'K18']
female30 = results['Finish_time[m]'][results['Cat'] == 'K30']
female40 = results['Finish_time[m]'][results['Cat'] == 'K40']
female45 = results['Finish_time[m]'][results['Cat'] == 'K45']
female50 = results['Finish_time[m]'][results['Cat'] == 'K50']
female55 = results['Finish_time[m]'][results['Cat'] == 'K55']
female60 = results['Finish_time[m]'][results['Cat'] == 'K60']
female65 = results['Finish_time[m]'][results['Cat'] == 'K65']
female70 = results['Finish_time[m]'][results['Cat'] == 'K70']


trace1 = go.Histogram(
    x=female18,
    name='F18',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx=False,
    marker=dict(
        color='#1e1c27'
    ),
)
trace2 = go.Histogram(
    x=female30,
    name='F30',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx=False,
    marker=dict(
        color='#3d394f'
    ),
)
trace3 = go.Histogram(
    x=female40,
    name='F40',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx=False,
    marker=dict(
        color='#5c5676'
    ),
)
trace4 = go.Histogram(
    x=female45,
    name='F45',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx=False,
    marker=dict(
        color='#7b739e'
    ),
)
trace5 = go.Histogram(
    x=female50,
    name='F50',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size= 'M33'
    ),
    autobinx = False,
    marker=dict(
        color='#9a90c6'
    ),
)
trace6 = go.Histogram(
    x=female55,
    name='F55',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx = False,
    marker=dict(
        color='#aea6d1'
    ),
)
trace7 = go.Histogram(
    x=female60,
    name='F60',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx=False,
    marker=dict(
        color='#c2bcdc'
    ),
)
trace8 = go.Histogram(
    x=female65,
    name='F65',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx=False,
    marker=dict(
        color='#d6d2e8'
    ),
)
trace0 = go.Histogram(
    x=female70,
    name='F70',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx=False,
    marker=dict(
        color='#eae8f3'
    ),
)

data = [trace1, trace2, trace3, trace4,
        trace5, trace6, trace7, trace8,
        trace0]

layout = go.Layout(
    title='Finish line time vs. # of women runners.',
    titlefont=dict(
        size=20,
        family='Droid Serif',
        color='#3B5F6D',
    ),
    barmode='stack',
    yaxis=dict(
        title='No. of runners',
        titlefont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D'
        ),
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='rgb(243, 243, 243)'
    ),
    xaxis=dict(
        title='Finish time [minutes]',
        titlefont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        tickmode='array',
        tickvals=[150, 180, 210, 240, 270, 300, 330, 360, 390],
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='rgb(243, 243, 243)',
    ),
    legend=dict(
        orientation='v',
        font=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D'
        ),
        x=0.92,
        y=1,
    ),
    shapes=[
        dict(
            layer='below',
            type='line',
            x0=eval(str(round(f_median, 2))),
            y0=20,
            x1=eval(str(round(f_median, 2))),
            y1=0,
            line=dict(
                color='#335b66',
                width=1,
            ),
        ),
    ],
    annotations=[
        dict(
            visible=True,
            x=266,
            y=19,
            xref='x',
            yref='y',
            text='median',
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
    ],
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
# Stacked men
male18 = results['Finish_time[m]'][results['Cat'] == 'M18']
male30 = results['Finish_time[m]'][results['Cat'] == 'M30']
male40 = results['Finish_time[m]'][results['Cat'] == 'M40']
male45 = results['Finish_time[m]'][results['Cat'] == 'M45']
male50 = results['Finish_time[m]'][results['Cat'] == 'M50']
male55 = results['Finish_time[m]'][results['Cat'] == 'M55']
male60 = results['Finish_time[m]'][results['Cat'] == 'M60']
male65 = results['Finish_time[m]'][results['Cat'] == 'M65']
male70 = results['Finish_time[m]'][results['Cat'] == 'M70']
male75 = results['Finish_time[m]'][results['Cat'] == 'M75']


trace0 = go.Histogram(
    x=male18,
    name='M18',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx=False,
    marker=dict(
        color='#1c2527'
    ),
)
trace1 = go.Histogram(
    x=male30,
    name='M30',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx=False,
    marker=dict(
        color='#394b4f'
    ),
)
trace2 = go.Histogram(
    x=male40,
    name='M40',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx=False,
    marker=dict(
        color='#567076'
    ),
)
trace3 = go.Histogram(
    x=male45,
    name='M45',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx=False,
    marker=dict(
        color='#73969e'
    ),
)
trace4 = go.Histogram(
    x=male50,
    name='M50',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size= 'M33'
    ),
    autobinx = False,
    marker=dict(
        color='#90bcc6'
    ),
)
trace5 = go.Histogram(
    x=male55,
    name='M55',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx = False,
    marker=dict(
        color='#9bc2cb'
    ),
)
trace6 = go.Histogram(
    x=male60,
    name='M60',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx=False,
    marker=dict(
        color='#b1d0d7'
    ),
)
trace7 = go.Histogram(
    x=male65,
    name='M65',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx=False,
    marker=dict(
        color='#c7dde2'
    ),
)
trace8 = go.Histogram(
    x=male70,
    name='M70',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx=False,
    marker=dict(
        color='#ddeaed'
    ),
)
trace9 = go.Histogram(
    x=male75,
    name='M75',
    opacity=0.6,
    xbins=dict(
        start=130,
        end=420,
        size='M33'
    ),
    autobinx=False,
    marker=dict(
        color='#e8f1f3'
    ),
)

data = [trace0, trace1, trace2, trace3, trace4,
        trace5, trace6, trace7, trace8, trace9]

layout = go.Layout(
    title='Finish line time vs. # of men runners.',
    titlefont=dict(
        size=20,
        family='Droid Serif',
        color='#3B5F6D',
    ),
    barmode='stack',
    yaxis=dict(
        title='No. of runners',
        titlefont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D'
        ),
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='rgb(243, 243, 243)'
    ),
    xaxis=dict(
        title='Finish time [minutes]',
        titlefont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        tickmode='array',
        tickvals=[150, 180, 210, 240, 270, 300, 330, 360, 390],
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='rgb(243, 243, 243)',
    ),
    legend=dict(
        orientation='v',
        font=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D'
        ),
        x=0.92,
        y=1,
    ),
    shapes=[
        dict(
            layer='below',
            type='line',
            x0=eval(str(round(m_median, 2))),
            y0=70,
            x1=eval(str(round(m_median, 2))),
            y1=0,
            line=dict(
                color='#335b66',
                width=1,
            ),
        ),
    ],
    annotations=[
        dict(
            visible=True,
            x=254,
            y=66,
            xref='x',
            yref='y',
            text='median',
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
    ],
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
#Normalized histogram
trace0 = go.Histogram(
    x=x0,
    name='M',
    xbins=dict(
        start=130,
        end=420,
        size= 'M33'),
    autobinx = False,
    opacity=0.5,
    marker=dict(
        color='#90BCC6'
    ),
    histnorm='percent',
    orientation='v',
)
trace1 = go.Histogram(
    x=x1,
    name='F',
    xbins=dict(
        start=130,
        end=420,
        size= 'M33'),
    autobinx = False,
    opacity=0.5,
    marker=dict(
        color='#9a90c6'
    ), 
    histnorm='percent',
    orientation='v',
)

layout=go.Layout(
    title='Finish line time vs. % # of runners.',
    titlefont=dict(
        size=20,
        family='Droid Serif',
        color='#3B5F6D',
    ),
    barmode='overlay',
    yaxis=dict(
        title='%',
        titlefont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D'
        ),
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='rgb(243, 243, 243)'
    ),
    xaxis=dict(
        title='Finish time [minutes]',
        titlefont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        tickmode='array',
        tickvals=[150, 180, 210, 240, 270, 300, 330, 360, 390],
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='rgb(243, 243, 243)',
    ),
    legend=dict(
        orientation='h',
        font=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D'
        ),
        x=0.8,
        y=0.94,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

data = [trace0, trace1]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
# Distplot with normal distribution
data = [x0, x1]
labels = ['M', 'F']
colors = ['#90BCC6', '#9a90c6']

fig = ff.create_distplot(data, labels, show_hist=False, curve_type='normal', colors=colors)

fig['layout'].update(
    title='Normal distribution curve',
    titlefont=dict(
        size=20,
        family='Droid Serif',
        color='#3B5F6D',
    ),
    yaxis=dict(
        titlefont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D'
        ),
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='rgb(243, 243, 243)'
    ),
    xaxis=dict(
        title='Finish time [minutes]',
        titlefont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        tickmode='array',
        tickvals=[150, 180, 210, 240, 270, 300, 330, 360, 390],
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='rgb(243, 243, 243)',
    ),
    legend=dict(
        orientation='v',
        font=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D'
        ),
        x=0.83,
        y=0.89,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)


py.iplot(fig)
# Halves
half_results = results[['Sex', '21.1KM', 'Finish','Finish_time[m]']]
half_results['delta_1'] = 0
half_results['delta_2'] = 0
half_results['%_diff'] = 0

def find_halves_diff(df):
    for i in range(0, len(df)):
        end_1 = pendulum.parse(df.iloc[i,1])
        end_2 = pendulum.parse(df.iloc[i,2])
        d_1 = end_1 - start
        d_2 = end_2 - end_1
        df.iloc[i,4] = d_1.total_minutes()
        df.iloc[i,5] = d_2.total_minutes()
        df.iloc[i,6] = ((df.iloc[i,4] - df.iloc[i,5]) / df.iloc[i,4])*100
    return df

half_results = find_halves_diff(half_results)  
# half_results[:5]
#Scatter plot
trace0 = go.Scatter(
    x = half_results['Finish_time[m]'][half_results['Sex']=='K'],
    y = half_results['%_diff'][half_results['Sex']=='K'],
    name = 'F',
    mode = 'markers',
    marker = dict(
        opacity=0.35,
        size = 9,
        color = '#9a90c6',
    ),
    hoverinfo='y',
)

trace1 = go.Scatter(
    x = half_results['Finish_time[m]'][half_results['Sex'] != 'K'],
    y = half_results['%_diff'][half_results['Sex'] != 'K'],
    name = 'M',
    mode = 'markers',
    marker = dict(
        size = 9,
        opacity=0.35,
        color = '#90BCC6',
    ),
    hoverinfo='y',
   
)

data = [trace1, trace0]

layout = dict(
    title = 'Finish line time vs. % difference between halves',
    titlefont=dict(
        size=20,
        family='Droid Serif',
        color='#3B5F6D',
    ),
    yaxis=dict(
        showgrid=False,
        zerolinecolor='#3B5F6D',
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='rgb(243, 243, 243)',
        ticksuffix=' %',
        showticksuffix='first'
    ),
    xaxis=dict(
        title='Finish time [minutes]',
        titlefont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        showgrid=False,
        tickmode='array',
        tickvals=[150, 180, 210, 240, 270, 300, 330, 360, 390],
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='rgb(243, 243, 243)',
        ticksuffix=' mins',
        showticksuffix='first'
    ),
    legend=dict(
        orientation='v',
        font=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D'
        ),
        x=0.83,
        y=0.89,
    ),
    annotations=[
        dict(
            visible=True,
            x=145,
            y=50,
            xref='x',
            yref='y',
            text='second half faster',
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
        dict(
            visible=True,
            x=145,
            y=-50,
            xref='x',
            yref='y',
            text='second half slower',
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
    ],
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

fig = dict(data=data, layout=layout)
py.iplot(fig)
# Calculate %s
second_slower_all = (half_results['%_diff'][half_results['%_diff']<0].count())/len(half_results)
second_slower_men = (half_results['%_diff'][(half_results['%_diff']<0) & (half_results['Sex']!='K')]
                    .count())/len(half_results['Sex'][half_results['Sex']!='K'])
second_slower_women = (half_results['%_diff'][(half_results['%_diff']<0) & (half_results['Sex']=='K')]
                    .count())/len(half_results['Sex'][half_results['Sex']=='K'])
second_faster_all = 1.0 - second_slower_all
second_faster_men = 1.0 - second_slower_men
second_faster_women = 1.0 - second_slower_women

print(f'{round(second_slower_all * 100, 1)}% of all runners run second half slower. Divided into\
 categories: {round(second_slower_men * 100, 1)}% of all men and {round(second_slower_women * 100, 1)}% \
of all women slowed down during second half of a distance.')#, second_slower_men, second_slower_women, second_faster_all, second_faster_men, second_faster_women)
# Make a df with paces between measurement points
sec_df = results[['Sex', '5KM', '10KM', '15KM',
                  '20KM', '21.1KM', '25KM',
                  '30KM', '35KM', '40KM', 'Finish']]
sec_dist = [5, 5, 5, 5, 1.092, 3.908, 5, 5, 5, 2.195]

def all_paces():
    for column in range(1, len(sec_df.columns)):
        for i in range(len(results)):
            end = pendulum.parse(results.iloc[i, column+5])
            if column==1:
                start = pendulum.datetime(2018, 10, 14, 9, 0, 0)
            else:
                start = pendulum.parse(results.iloc[i, column+4])
            delta = end - start
            sec_df.iloc[i,column] = (delta.total_minutes()/sec_dist[column-1])
    return sec_df

sec_df = all_paces()
sec_df[:5]
# Find average tempo for women and men between contorl points.
f_sec_tempo = []
m_sec_tempo = []

def find_mean_tempo():
    for column in range(1, len(sec_df.columns)):
        f_sec_tempo.append(sec_df.iloc[:,column][sec_df['Sex']=='K'].mean())
        m_sec_tempo.append(sec_df.iloc[:,column][sec_df['Sex']!='K'].mean())
    return f_sec_tempo, m_sec_tempo

f_sec_tempo, m_sec_tempo = find_mean_tempo()
print(f"Womens sections tempo {f_sec_tempo}, and mens sections tempo {m_sec_tempo}")
# Prepare df for plotting
l=list(sec_df)[1:11]
sec_df = sec_df[['5KM', '10KM', '15KM', '20KM', '21.1KM', '25KM',
                 '30KM', '35KM', '40KM', 'Finish']]

# Pace chart
traces = []

for row in sec_df.itertuples(index=False):
    traces.append(go.Scatter(
        x=l,
        y=list(row),
        mode='lines',
        line=dict(
            width=0.1, 
            color='black',
        ),
        opacity=0.2,
        hoverinfo='none'
    ))
    
    
layout = go.Layout(
    title='Runners pace on selected points',
    titlefont=dict(
        size=20,
        family='Droid Serif',
        color='#3B5F6D',
    ),
    xaxis=dict(
        showline=False,
        title='Time measurement point',
        titlefont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        showgrid=False,
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='#3B5F6D',
    ),
    yaxis=dict(
        range=[10,3],
        title='Pace [min/km]',
        titlefont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D'
        ),
        showgrid=False,
        zerolinecolor='#3B5F6D',
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='#3B5F6D',
    ),
    showlegend=False,
    shapes=[
        dict(
            layer='below',
            type='line',
            x0='5KM',
            y0=6.043,
            x1='Finish',
            y1=6.043,
            line=dict(
                color='#ECA400',
                width=1,
            ),
        ),
        dict(
            layer='below',
            type='line',
            x0='5KM',
            y0=5.688,
            x1='Finish',
            y1=5.688,
            line=dict(
                color='#ECA400',
                width=1,
            ),
        ),
        dict(
            layer='below',
            type='line',
            x0='5KM',
            y0=4.967,
            x1='Finish',
            y1=4.967,
            line=dict(
                color='#ECA400',
                width=1,
            ),
        ),
        dict(
            layer='below',
            type='line',
            x0='5KM',
            y0=4.266,
            x1='Finish',
            y1=4.266,
            line=dict(
                color='#ECA400',
                width=1,
            ),
        ),
    ],
    annotations=[
        dict(
            visible=True,
            x=8.5,
            y=4.116,
            xref='x',
            yref='y',
            text='3:00h',
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
        dict(
            visible=True,
            x=8.5,
            y=4.817,
            xref='x',
            yref='y',
            text='3:30h',
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
        dict(
            visible=True,
            x=8.5,
            y=5.538,
            xref='x',
            yref='y',
            text='4:00h',
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
        dict(
            visible=True,
            x=8.5,
            y=5.893,
            xref='x',
            yref='y',
            text='4:15h',
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
    ],
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)
fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)
# Add column with the net start time

def add_net_start_time(df):
    df['start_time_net'] = 0
    for i in range(0, len(df)):
        net_time = pendulum.parse(df.iloc[i,16], exact=True)
        gross_time = pendulum.parse(df.iloc[i,17], exact=True)
        delta_net_gross = gross_time - net_time
        df.iloc[i,19] = delta_net_gross
    return df

results = add_net_start_time(results)
#print(results['start_time_net'][:5])
# Prepare data to plot new pace
results_net = results[['5KM', '10KM', '15KM','20KM', '25KM',
                       '30KM', '35KM', '40KM', 'Finish', 'start_time_net']]
results_dist = [5, 5, 5, 5, 5, 5, 5, 5, 2.195]

results_net_copy = results_net.copy() # to avoid overwrite data in results_net
# Create df with new net paces

def all_paces_net():
    for column in range(len(results_net_copy.columns)-1):
        for i in range(len(results_net_copy)):
            end = pendulum.parse(results_net.iloc[i, column])
            if column==0:
                start = pendulum.datetime(2018, 10, 14, 9, 0, 0) + results_net['start_time_net'][i]
            else:
                start = pendulum.parse(results_net.iloc[i, column-1])
            delta = end - start
            results_net_copy.iloc[i,column] = (delta.total_minutes()/results_dist[column])
    return results_net_copy

results_net_copy = all_paces_net()
#results_net_copy[:5]
# Pace chart
traces = []

results_net_copy = results_net_copy[['5KM', '10KM', '15KM', '20KM', '25KM',
                                     '30KM', '35KM', '40KM', 'Finish']]
l = [5, 10, 15, 20, 25, 30, 35, 40, 42.195]
labels = ['5KM', '10KM', '15KM', '20KM',
          '25KM', '30KM', '35KM', '40KM', 'Finish']


for row in results_net_copy.itertuples(index=False):
    traces.append(go.Scatter(
        x=l,
        y=list(row),
        mode='lines',
        line=dict(
            width=0.1, 
            color='black',
        ),
        opacity=0.2,
        hoverinfo='none'
    ))
    
    
layout = go.Layout(
    title='Runners pace on selected points',
    titlefont=dict(
        size=20,
        family='Droid Serif',
        color='#3B5F6D',
    ),
    xaxis=dict(
        showline=False,
        title='Time measurement point',
        titlefont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        showgrid=False,
        tickvals=l,
        ticktext=labels,
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='#3B5F6D',
    ),
    yaxis=dict(
        range=[10,3],
        title='Pace [min/km]',
        titlefont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D'
        ),
        showgrid=False,
        zerolinecolor='#3B5F6D',
        tickfont=dict(
            size=14,
            family='Droid Serif',
            color='#3B5F6D',
        ),
        ticks='outside',
        ticklen=8,
        tickcolor='#3B5F6D',
    ),
    showlegend=False,
    shapes=[
        dict(
            layer='below',
            type='line',
            x0=5,
            y0=6.043,
            x1=44,
            y1=6.043,
            line=dict(
                color='#ECA400',
                width=1,
            ),
        ),
        dict(
            layer='below',
            type='line',
            x0=5,
            y0=5.688,
            x1=44,
            y1=5.688,
            line=dict(
                color='#ECA400',
                width=1,
            ),
        ),
        dict(
            layer='below',
            type='line',
            x0=5,
            y0=4.967,
            x1=44,
            y1=4.967,
            line=dict(
                color='#ECA400',
                width=1,
            ),
        ),
        dict(
            layer='below',
            type='line',
            x0=5,
            y0=4.266,
            x1=44,
            y1=4.266,
            line=dict(
                color='#ECA400',
                width=1,
            ),
        ),
    ],
    annotations=[
        dict(
            visible=True,
            x=43,
            y=4.116,
            xref='x',
            yref='y',
            text='3:00h',
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
        dict(
            visible=True,
            x=43,
            y=4.817,
            xref='x',
            yref='y',
            text='3:30h',
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
        dict(
            visible=True,
            x=43,
            y=5.538,
            xref='x',
            yref='y',
            text='4:00h',
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
        dict(
            visible=True,
            x=43,
            y=5.893,
            xref='x',
            yref='y',
            text='4:15h',
            ax=0,
            ay=0,
            font=dict(
                size=12,
                family='Droid Serif',
                color='#3B5F6D',
            ),
        ),
    ],
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)
fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)