# import the usual frameworks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import warnings

from IPython.core.display import display, HTML
from sklearn.preprocessing import MinMaxScaler

import os
print(os.listdir("../input"))
    
# import plotly 
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls

# for color scales in plotly
import colorlover as cl 

# define color scale https://plot.ly/ipython-notebooks/color-scales/
cs = cl.scales['10']['div']['RdYlGn']    # for most charts 
cs7 =  cl.scales['7']['qual']['Dark2']   # for stacked bar charts  

# configure things
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:,.2f}'.format  
pd.options.display.max_columns = 999

py.init_notebook_mode(connected=True)

%load_ext autoreload
%autoreload 2
%matplotlib inline
#!pip list
new_col_names = ['framework','indeed', 'monster', 'simply', 'linkedin', 'angel', 
                 'usage', 'search', 'medium', 'books', 'arxiv', 'stars', 
                 'watchers', 'forks', 'contribs',
                ]

df = pd.read_csv('../input/ds13.csv', 
                 skiprows=4,
                 header=None, 
                 nrows=11, 
                 thousands=',',
                 index_col=0,
                 names=new_col_names,
                 usecols=new_col_names,
                )
df
df.info()
df.describe()
df['usage'] = pd.to_numeric(df['usage'].str.strip('%'))
df['usage'] = df['usage'].astype(int)
df
df.info()
# sum groupby for the hiring columns
df['hiring'] = df['indeed'] + df['monster'] + df['linkedin'] + df['simply'] + df['angel']
df
data = [go.Bar(
    x=df.index,
    y=df.hiring,
    marker=dict(color=cs),
    )
]

layout = {'title': 'Online Job Listings',
          'xaxis': {'title': 'Framework'},
          'yaxis': {'title': "Quantity"},
         }

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
y_indeed = df['indeed']
y_monster = df['monster']
y_simply = df['simply']
y_linkedin = df['linkedin']
y_angel = df['angel']
indeed = go.Bar(x=df.index, y=y_indeed, name = 'Indeed')
simply = go.Bar(x=df.index, y=y_simply, name='Simply Hired')
monster = go.Bar(x=df.index, y=y_monster, name='Monster')
linked = go.Bar(x=df.index, y=y_linkedin, name='LinkedIn')
angel = go.Bar(x=df.index, y=y_angel, name='Angel List')

data = [linked, indeed, simply, monster, angel]
layout = go.Layout(
    barmode='stack',
    title='Online Job Listings',
    xaxis={'title': 'Framework'},
    yaxis={'title': 'Mentions', 'separatethousands': True},
    colorway=cs,
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
indeed = go.Bar(x=df.index, y=y_indeed, name = "Indeed")
simply = go.Bar(x=df.index, y=y_simply, name="Simply Hired")
monster = go.Bar(x=df.index, y=y_monster, name="Monster")
linked = go.Bar(x=df.index, y=y_linkedin, name="LinkedIn")
angel = go.Bar(x=df.index, y=y_angel, name='Angel List')

data = [linked, indeed, simply, monster, angel]
layout = go.Layout(
    barmode='group',
    title="Online Job Listings",
    xaxis={'title': 'Framework'},
    yaxis={'title': "Listings", 'separatethousands': True,
    }
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
# Make sure you have colorlover imported as cl for color scale
df['usage'] = df['usage'] / 100
data = [
    go.Bar(
        x=df.index, 
        y=df['usage'],
        marker=dict(color=cs)  
    )
]
    
layout = {
    'title': 'KDnuggets Usage Survey',
    'xaxis': {'title': 'Framework'},
    'yaxis': {'title': "% Respondents Used in Past Year", 'tickformat': '.0%'},
}

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
data = [
    go.Bar(
        x = df.index, 
        y = df['search'],
        marker = dict(color=cs),  
    )
]
    
layout = {
    'title': 'Google Search Volume',
    'xaxis': {'title': 'Framework'},
    'yaxis': {'title': "Relative Search Volume"},
}

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
# Make sure you have colorlover imported as cl for color scale
# cs is defined in first cell

data = [
    go.Bar(
        x=df.index, 
        y=df['medium'],
        marker=dict(color=cs) ,
    )
]
    
layout = {
    'title': 'Medium Articles',
    'xaxis': {'title': 'Framework'},
    'yaxis': {'title': "Articles"},
}

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
data = [
    go.Bar(
        x=df.index, 
        y=df['books'],
        marker=dict(color=cs),           
    )
]
    
layout = {
    'title': 'Amazon Books',
    'xaxis': {'title': 'Framework'},
    'yaxis': {'title': "Books"},
}

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
data = [
    go.Bar(
        x=df.index, 
        y=df['arxiv'],
        marker=dict(color=cs),           
    )
]

layout = {
    'title': 'ArXiv Articles',
    'xaxis': {'title': 'Framework'},
    'yaxis': {'title': "Articles"},
}

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
y_stars = df['stars']
y_watchers = df['watchers']
y_forks = df['forks']
y_contribs = df['contribs']

stars = go.Bar(x = df.index, y=y_stars, name="Stars")
watchers = go.Bar(x=df.index, y=y_watchers, name="Watchers")
forks = go.Bar(x=df.index, y=y_forks, name="Forks")
contribs = go.Bar(x=df.index, y=y_contribs, name="Contributors")


data = [stars, watchers, forks, contribs]
layout = go.Layout(barmode='stack', 
    title="GitHub Activity",
    xaxis={'title': 'Framework'},
    yaxis={
        'title': "Quantity",
        'separatethousands': True,
    }
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
trace1 = go.Bar(
    x=df.index,
    y=df['stars'],
    name='Stars',
    marker=dict(color=cs),
)
trace2 = go.Bar(
    x=df.index,
    y=df['forks'],
    name ="Forks",
    marker=dict(color=cs)
)
trace3 = go.Bar(
    x=df.index,
    y=df['watchers'],
    name='Watchers',
    marker=dict(color=cs)
)
trace4 = go.Bar(
    x=df.index,
    y=df['contribs'],
    name='Contributors',
    marker=dict(color=cs),
)

fig = tls.make_subplots(
    rows=2, 
    cols=2, 
    subplot_titles=(
        'Stars', 
        'Forks',
        'Watchers',
        'Contributors',
    )
)

fig['layout']['yaxis3'].update(separatethousands = True)
fig['layout']['yaxis4'].update(separatethousands = True)
fig['layout']['yaxis2'].update(tickformat = ',k', separatethousands = True)
fig['layout']['yaxis1'].update(tickformat = ',k', separatethousands = True)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)

fig['layout'].update(title = 'GitHub Activity', showlegend = False)
py.iplot(fig)
df.info()
scale = MinMaxScaler()
scaled_df = pd.DataFrame(
    scale.fit_transform(df), 
    columns = df.columns,
    index = df.index)    
scaled_df
scaled_df['hiring_score'] = scaled_df[['indeed', 'monster', 'simply', 'linkedin', 'angel']].mean(axis=1)
scaled_df
scaled_df['github_score'] = scaled_df[['stars', 'watchers', 'forks', 'contribs']].mean(axis=1)
scaled_df
weights = {'Online Job Listings ': .3,
           'KDnuggets Usage Survey': .2,
           'GitHub Activity': .1,
           'Google Search Volume': .1,
           'Medium Articles': .1,
           'Amazon Books': .1,
           'ArXiv Articles': .1 }
# changing colors because we want to show these aren't the frameworks
weight_colors = cl.scales['7']['qual']['Set1'] 

common_props = dict(
    labels = list(weights.keys()),
    values = list(weights.values()),
    textfont=dict(size=16),
    marker=dict(colors=weight_colors),
    hoverinfo='none',
    showlegend=False,
)

trace1 = go.Pie(
    **common_props,
    textinfo='label',
    textposition='outside',
)

trace2 = go.Pie(
    **common_props,
    textinfo='percent',
    textposition='inside',
)

layout = go.Layout(title = 'Weights by Category')

fig = go.Figure([trace1, trace2], layout=layout)
py.iplot(fig)
scaled_df['w_hiring'] = scaled_df['hiring_score'] * .3
scaled_df['w_usage'] = scaled_df['usage'] * .2
scaled_df['w_github'] = scaled_df['github_score'] * .1
scaled_df['w_search'] = scaled_df['search'] * .1
scaled_df['w_arxiv'] = scaled_df['arxiv'] * .1
scaled_df['w_books'] = scaled_df['books'] * .1
scaled_df['w_medium'] = scaled_df['medium'] * .1
weight_list = ['w_hiring', 'w_usage', 'w_github', 'w_search', 'w_arxiv', 'w_books', 'w_medium']
scaled_df = scaled_df[weight_list]
scaled_df
scaled_df['ps'] = scaled_df[weight_list].sum(axis = 1)
scaled_df
p_s_df = scaled_df * 100
p_s_df = p_s_df.round(2)
p_s_df.columns = ['Job Search Listings', 'Usage Survey', 'GitHub Activity', 'Search Volume', 'ArXiv Articles', 'Amazon Books', 'Medium Articles', 'Power Score']
p_s_df.rename_axis('Framework', inplace = True)
p_s_df
data = [
    go.Bar(
        x=scaled_df.index,          # you can pass plotly the axis
        y=p_s_df['Power Score'],
        marker=dict(color=cs),
        text=p_s_df['Power Score'],
        textposition='outside',
        textfont=dict(size=10)
    )
]

layout = {
    'title': 'Deep Learning Framework Power Scores 2018',
    'xaxis': {'title': 'Framework'},
    'yaxis': {'title': "Score"}
}

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
