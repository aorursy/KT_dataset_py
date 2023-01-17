# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import math

from bubbly.bubbly import add_slider_steps

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

from plotly import tools

import plotly.figure_factory as ff 

import warnings

warnings.filterwarnings('ignore')

import os



data = pd.read_csv("../input/article_data.csv")

data.columns = ['indice', 'date', 'n1146295', 'n1652858', 'n1652930', 'n3757016',

       'n4509650', 'n3757022', 'n4509627', 'n4509607', 'n3757020', 'n4509636',

       'n3757019', 'n3757018', 'n3757021', 'n3757017', 'n1653018', 'n1653100',

       'n3767755', 'character', 'n1146557', 'n1146624', 'n1146592', 'n3917132',

       'n4545264', 'n1146562', 'n1146650', 'n1146655', 'n1146582',

       'Important_cCodes', 'n_posts_that_day', 'day', 'week', 'month', 'year',

       'weekdays', 'QTR', 'doc_release', 'policy_explained', 'topic_1', 'topic_2', 'topic_3',

       'topic_4', 'topic_5']



agg_df1 = pd.DataFrame(data.groupby(['month','year']).size())

agg_df2 = pd.DataFrame(data.groupby(['month','year'])[['topic_1','topic_2','topic_3','topic_4','topic_5']].median())

AGG = pd.DataFrame(agg_df1.join(agg_df2))

AGG.columns = ['count','topic_1','topic_2', 'topic_3','topic_4', 'topic_5']

AGG = AGG.reset_index(inplace = False) ### MUST INPLACE FALSE

AGG = AGG.sort_values(by=['year']).reset_index(drop=True)

AGG['majority_topic'] = AGG[['topic_1','topic_2','topic_3','topic_4','topic_5']].idxmax(axis=1)

data.tail()
import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import pandas as pd

import math



data = pd.read_csv("../input/article_data.csv")

data.columns = ['indice', 'date', 'n1146295', 'n1652858', 'n1652930', 'n3757016',

       'n4509650', 'n3757022', 'n4509627', 'n4509607', 'n3757020', 'n4509636',

       'n3757019', 'n3757018', 'n3757021', 'n3757017', 'n1653018', 'n1653100',

       'n3767755', 'character', 'n1146557', 'n1146624', 'n1146592', 'n3917132',

       'n4545264', 'n1146562', 'n1146650', 'n1146655', 'n1146582',

       'Important_cCodes', 'n_posts_that_day', 'day', 'week', 'month', 'year',

       'weekdays', 'QTR', 'doc_release', 'policy_explained', 'topic_1', 'topic_2', 'topic_3',

       'topic_4', 'topic_5']





EX = pd.DataFrame(data.groupby('year')[['topic_1','topic_2','topic_3','topic_4','topic_5']].median()).reset_index(inplace=False)





x= EX.year



trace1 = dict(

    x= x,

    y= data['topic_1'],

    name='Industrial & Business Information Products',

    hoverinfo='x+y',

    mode='lines',

    line=dict(width=0.5),

              #color='rgb(131, 90, 241)'),

    stackgroup='one'

)

trace2 = dict(

    x= x,

    y= data['topic_2'],

    name='Public Opinion concerning Industrial Policy',

    hoverinfo='x+y',

    mode='lines',

    line=dict(width=0.5),

              #color='rgb(111, 231, 219)'),

    stackgroup='one'

)

trace3 = dict(

    x= x,

    y= data['topic_3'],

    name='Internet Product Marketization',

    hoverinfo='x+y',

    mode='lines',

    line=dict(width=0.5),

              #color='rgb(184, 247, 212)'),

    stackgroup='one'

)

trace4 = dict(

    x= x,

    y= data['topic_4'],

    name='Automotive Production & Equipment',

    hoverinfo='x+y',

    mode='lines',

    line=dict(width=0.5),

              #color='rgb(184, 247, 212)'),

    stackgroup='one'

)

trace5 = dict(

    x= x,

    y= data['topic_5'],

    name='Setting National Business Tech Standards',

    hoverinfo='x+y',

    mode='lines',

    line=dict(width=0.5),

              #color='rgb(243, 243, 243)'),

    stackgroup='one'

)



data = [trace1, trace2, trace3, trace4, trace5]



layout = go.Layout(

    

    title=go.layout.Title(

        text='Figure 1: Shift of Government Priorities by Policy Topics by MIIT (2002-2018)',

        xref='paper',

        #x=0,

        font=dict( size=22)

    ),



#     xaxis=go.layout.XAxis(

#         title=go.layout.xaxis.Title(

#             text='Year',

#             font=dict(

#                 #family='monospace',

#                 size=20,

#                 #color='#7f7f7f'

#             )

#         )

        

#     ),

    yaxis=go.layout.YAxis(

        title=go.layout.yaxis.Title(

            text='Percent of Topical Distribution',

            font=dict(

                #family='monospace',

                size=20,

                #color='#7f7f7f'

            )

        )

    ),

    legend=dict(x=0, y=-.13,

       orientation="h")

)

fig = go.Figure(data=data, layout=layout)

iplot(fig) 

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)





dataset = AGG.sort_values(['year', 'month'], ascending=[True, True])

x_column = 'month'

x_column2 = 'topic_2'

y_column = 'count'

bubble_column = 'count'

time_column = 'year'

size_column = 'count'

# 2002, 2003, 2004, 2005, 2006, 2007,

years = np.array([2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018])

### GRID

grid = pd.DataFrame()

col_name_template = '{year}+{header}_grid'

for year in years:

    dataset_by_year = dataset[(dataset['year'] == int(year))]

    for col_name in [x_column, y_column, bubble_column]:

        # Each column name is unique

        temp = col_name_template.format(

            year=year, header=col_name

        )

        if dataset_by_year[col_name].size != 0:

            grid = grid.append({'value': list(dataset_by_year[col_name]), 'key': temp}, 

                               ignore_index=True)



            

# Define figure

figure = {

    'data': [],

    'layout': {},

    'frames': []

}



# Get the earliest year

year = min(years)



# Make the trace

trace = {

    'x': grid.loc[grid['key']==col_name_template.format(

        year=year, header=x_column

    ), 'value'].values[0], 

    'y': grid.loc[grid['key']==col_name_template.format(

        year=year, header=y_column

    ), 'value'].values[0],

    #'mode': 'markers',

    'text': grid.loc[grid['key']==col_name_template.format(

        year=year, header=bubble_column

    ), 'value'].values[0],

    'type': 'scatter'

}

# Append the trace to the figure

figure['data'].append(trace)            

# Get the max and min range of both axes

xmin = min(dataset[x_column])

xmax = max(dataset[x_column])

ymin = min(dataset[y_column])

ymax = max(dataset[y_column])



# Modify the layout

figure['layout']['xaxis'] = {'title': 'Month', 

                             'range': [0, 12.5],

                             'dtick': 1}   

figure['layout']['yaxis'] = {'title': 'Number of Policy Documents', 

                             'range': [ymin, ymax]} 

figure['layout']['title'] = 'Figure 2: Historic Yearly MIIT Policy Output Frequency (2002-2018)'

figure['layout']['showlegend'] = False

figure['layout']['hovermode'] = 'closest'



######################################### ADD ANIMATED TIME FRAMES

for year in years:

    # Make a frame for each year

    frame = {'data': [], 'name': str(year)}

    

    # Make a trace for each frame

    trace = {

        'x': grid.loc[grid['key']==col_name_template.format(

            year=year, header=x_column

        ), 'value'].values[0],

        'y': grid.loc[grid['key']==col_name_template.format(

            year=year, header=y_column

        ), 'value'].values[0],

        #'mode': 'mark',

        'text': grid.loc[grid['key']==col_name_template.format(

            year=year, header=bubble_column

        ), 'value'].values[0],

        'type': 'scatter'

    }

    # Add trace to the frame

    frame['data'].append(trace)

    # Add frame to the figure

    figure['frames'].append(frame) 



######################################### ADD SLIDERS

figure['layout']['sliders'] = {

    'args': [

        'slider.value', {

            'duration': 400,

            'ease': 'cubic-in-out'

        }

    ],

    'initialValue': min(years),

    'plotlycommand': 'animate',

    'values': years,

    'visible': True

}

sliders_dict = {

    'active': 0,

    'yanchor': 'top',

    'xanchor': 'left',

    'currentvalue': {

        'font': {'size': 20},

        'prefix': 'Year:',

        'visible': True,

        'xanchor': 'right'

    },

    'transition': {'duration': 300, 'easing': 'cubic-in-out'},

    'pad': {'b': 10, 't': 50},

    'len': 0.9,

    'x': 0.1,

    'y': 0,

    'steps': []

}



from bubbly.bubbly import add_slider_steps



for year in years:

    add_slider_steps(sliders_dict, year)

    

figure['layout']['sliders'] = [sliders_dict]

##################################### Add Play and Pause Buttons



figure['layout']['updatemenus'] = [

    {

        'buttons': [

            {

                'args': [None, {'frame': {'duration': 500, 'redraw': True},

                         'fromcurrent': True, 'transition': {'duration': 300, 

                                                             'easing': 'quadratic-in-out'}}],

                'label': 'Play',

                'method': 'animate'

            },

            {

                'args': [[None], {'frame': {'duration':0, 'redraw': False}, 'mode': 'immediate',

                'transition': {'duration': 0}}],

                'label': 'Pause',

                'method': 'animate'

            }

        ],

        'direction': 'left',

        'pad': {'r': 10, 't': 87},

        'showactive': False,

        'type': 'buttons',

        'x': 0.1,

        'xanchor': 'right',

        'y': 0,

        'yanchor': 'top'

    }

]



py.iplot(figure, config={'scrollzoom': True})
import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import pandas as pd

import math



data = pd.read_csv("../input/article_data.csv")

data.columns = ['indice', 'date', 'n1146295', 'n1652858', 'n1652930', 'n3757016',

       'n4509650', 'n3757022', 'n4509627', 'n4509607', 'n3757020', 'n4509636',

       'n3757019', 'n3757018', 'n3757021', 'n3757017', 'n1653018', 'n1653100',

       'n3767755', 'character', 'n1146557', 'n1146624', 'n1146592', 'n3917132',

       'n4545264', 'n1146562', 'n1146650', 'n1146655', 'n1146582',

       'Important_cCodes', 'n_posts_that_day', 'day', 'week', 'month', 'year',

       'weekdays', 'QTR', 'doc_release', 'policy_explained', 'topic_1', 'topic_2', 'topic_3',

       'topic_4', 'topic_5']



DATA = pd.DataFrame(data[data['year']==2017])

DATA['date'] = DATA['date'].astype('datetime64[ns]')

EX = pd.DataFrame(DATA.groupby('date')[['topic_1','topic_2','topic_3','topic_4','topic_5']].mean()).reset_index(inplace=False)





trace1 = dict(

    x= EX['date'],

    y= EX['topic_1'],

    name='Industrial & Business Information Products',

    hoverinfo='x+y',

    mode='lines',

    line=dict(width=0.5),

              #color='rgb(131, 90, 241)'),

    stackgroup='one'

)

trace2 = dict(

    x= EX['date'],

    y= EX['topic_2'],

    name='Public Opinion concerning Industrial Policy',

    hoverinfo='x+y',

    mode='lines',

    line=dict(width=0.5),

              #color='rgb(111, 231, 219)'),

    stackgroup='one'

)

trace3 = dict(

    x= EX['date'],

    y= EX['topic_3'],

    name='Internet Product Marketization',

    hoverinfo='x+y',

    mode='lines',

    line=dict(width=0.5),

              #color='rgb(184, 247, 212)'),

    stackgroup='one'

)

trace4 = dict(

    x= EX['date'],

    y= EX['topic_4'],

    name='Automotive Production & Equipment',

    hoverinfo='x+y',

    mode='lines',

    line=dict(width=0.5),

              #color='rgb(184, 247, 212)'),

    stackgroup='one'

)

trace5 = dict(

    x= EX['date'],

    y= EX['topic_5'],

    name='Setting National Business Standards',

    hoverinfo='x+y',

    mode='lines',

    line=dict(width=0.5),

              #color='rgb(243, 243, 243)'),

    stackgroup='one'

)



data = [trace1, trace2, trace3, trace4, trace5]



layout = go.Layout(



    title=go.layout.Title(

        text='Figure 3: Shift of Government Priorities by Policy Topics by MIIT in 2017',

        xref='paper',

        #x=0,

        font=dict( size=18)

    ),

    showlegend = True,



#     xaxis=go.layout.XAxis(

#         range = [min(EX.date),max(EX.date)],

#         title=go.layout.xaxis.Title(

#             text='Month',

#             font=dict(

#                 #family='monospace',

#                 size=20,

#                 #color='#7f7f7f'

#             )

#         )

        

#     ),

    yaxis=go.layout.YAxis(

        title=go.layout.yaxis.Title(

            text='Percent of Topical Distribution',

            font=dict(

                #family='monospace',

                size=18,

                #color='#7f7f7f'

            )

        )

    ),

    legend=dict(x=0, y=-.13,

       orientation="h")

)

fig = go.Figure(data=data, layout=layout)

iplot(fig) 
