# import the usual frameworks

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import collections

import warnings

import os



from IPython.core.display import display, HTML



from sklearn.preprocessing import MinMaxScaler



# plotly for charts

import plotly.graph_objs as go

import plotly.offline as py

import plotly.tools as tls

import colorlover as cl 



# define color scale https://plot.ly/ipython-notebooks/color-scales/

cs = cl.scales['4']['div']['RdYlGn']    # for most charts 

cs7 =  cl.scales['7']['qual']['Dark2']   # for stacked bar charts  



# configure things

warnings.filterwarnings('ignore')



#pd.options.display.float_format = '{:,.2f}'.format  



py.init_notebook_mode(connected=False)



%load_ext autoreload

%autoreload 2

%matplotlib inline
new_col_names = ['framework','indeed', 'monster', 'simply', 'linkedin', 

                 'google', 'medium', 'arxiv', 'quora', 

                 'stars', 'watchers', 'forks', 'contribs',

                ]



df = pd.read_csv('../input/additions_mar.csv', 

                 skiprows=1,

                 header=None, 

                 nrows=11, 

                 thousands=',',

                 index_col=0,

                 names=new_col_names,

                 usecols=new_col_names,

                )



df
df.info()
# sum groupby for the hiring columns

df['hiring'] = df['indeed'] + df['monster'] + df['linkedin'] + df['simply'] 
df
data = [

    go.Bar(

        x=df.index,

        y=df.hiring,

        marker=dict(color=cs),

    )

]



layout = go.Layout(

    dict(

        title='Online Job Listing Growth',

        xaxis=dict(title='Framework'),

        yaxis=dict(title='Change in Listings'),

    )

)



    

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
y_indeed = df['indeed']

y_monster = df['monster']

y_simply = df['simply']

y_linkedin = df['linkedin']

indeed = go.Bar(x=df.index, y=y_indeed, name = 'Indeed')

simply = go.Bar(x=df.index, y=y_simply, name='Simply Hired')

monster = go.Bar(x=df.index, y=y_monster, name='Monster')

linked = go.Bar(x=df.index, y=y_linkedin, name='LinkedIn')



data = [linked, indeed, simply, monster]

layout = go.Layout(

    barmode='stack',

    title='Online Job Listing Growth',

    xaxis={'title': 'Framework'},

    yaxis={'title': 'Change in Listings', 'separatethousands': True},

    colorway=cs,

)



fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
indeed = go.Bar(x=df.index, y=y_indeed, name = "Indeed")

simply = go.Bar(x=df.index, y=y_simply, name="Simply Hired")

monster = go.Bar(x=df.index, y=y_monster, name="Monster")

linked = go.Bar(x=df.index, y=y_linkedin, name="LinkedIn")





data = [linked, indeed, simply, monster]

layout = go.Layout(

    barmode='group',

    title="Online Job Listing Growth by Website",

    xaxis={'title': 'Framework'},

    yaxis={'title': "Listings", 'separatethousands': True,

    }

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
data = [

    go.Bar(

        x = df.index, 

        y = df['google'],

        marker = dict(color=cs),  

    )

]

    

layout = {

    'title': 'Google Search: Past 6 Months to Prior 6 Months',

    'xaxis': {'title': 'Framework'},

    'yaxis': {'title': "Average Search Interest"},

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

    'title': 'New Medium Articles',

    'xaxis': {'title': 'Framework'},

    'yaxis': {'title': "Articles"},

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

    'title': 'New arXiv Articles',

    'xaxis': {'title': 'Framework'},

    'yaxis': {'title': "Articles"},

}



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
data = [

    go.Bar(

        x=df.index, 

        y=df['quora'],

        marker=dict(color=cs),           

    )

]



layout = {

    'title': 'New Quora Topic Followers',

    'xaxis': {'title': 'Framework'},

    'yaxis': {'title': "Followers"},

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



fig['layout'].update(title = 'New GitHub Activity', showlegend = False)

py.iplot(fig)
df.info()
scale = MinMaxScaler()

scaled_df = pd.DataFrame(

    scale.fit_transform(df), 

    columns = df.columns,

    index = df.index)    
scaled_df
scaled_df['hiring_score'] = scaled_df[['indeed', 'monster', 'simply', 'linkedin']].mean(axis=1)
scaled_df
scaled_df['github_score'] = scaled_df[['stars', 'watchers', 'forks', 'contribs']].mean(axis=1)
scaled_df
weights = {'Online Job Listings ': .35,

           'Google Seach Interest': .13,

           'GitHub Activity': .13,

           'Quora Followers': .13,

           'Medium Articles': .13,

           'ArXiv Articles': .13 }
# changing colors because we want to show these aren't the frameworks

weight_colors = cl.scales['6']['qual']['Set1'] 



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
scaled_df['w_hiring'] = scaled_df['hiring_score'] * .35

scaled_df['w_google'] = scaled_df['google'] * .13

scaled_df['w_github'] = scaled_df['github_score'] * .13

scaled_df['w_arxiv'] = scaled_df['arxiv'] * .13

scaled_df['w_medium'] = scaled_df['medium'] * .13

scaled_df['w_quora'] = scaled_df['quora'] * .13
weight_list = ['w_hiring', 'w_google', 'w_github', 'w_arxiv',  'w_medium', 'w_quora']

scaled_df = scaled_df[weight_list]

scaled_df
scaled_df['gs'] = scaled_df[weight_list].sum(axis = 1)

scaled_df
g_s_df = scaled_df * 100

g_s_df = g_s_df.round(0)

g_s_df.columns = ['Job Search Listings', 'Google Interest', 'GitHub Activity',  'ArXiv Articles', 'Medium Articles', 'Quora Followers', 'Growth Score']

g_s_df.rename_axis('Framework', inplace = True)

g_s_df
data = [

    go.Bar(

        x=scaled_df.index,          

        y=g_s_df['Growth Score'],

        marker=dict(color=cs),

        text=g_s_df['Growth Score'],

        textposition='outside',

        textfont=dict(size=10)

    )

]



layout = {

    'title': 'Deep Learning Framework Six-Month Growth Scores 2019',

    'xaxis': {'title': 'Framework'},

    'yaxis': {'title': "Score"}

}



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)