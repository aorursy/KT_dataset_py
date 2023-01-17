# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected=True)  

import plotly.figure_factory as ff

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
super_bowl = pd.read_csv('../input/superbowl/Super_Bowl.csv')

super_bowl.head()
from datetime import timedelta, date

col = 'Date'

super_bowl[col] = pd.to_datetime(super_bowl[col])

future = super_bowl[col] > date(year=2050,month=1,day=1)

super_bowl.loc[future, col] -= timedelta(days=365.25*100)
super_bowl
super_bowl.info()
trace_high = go.Scatter(

    x=super_bowl.Date,

    y=super_bowl.Attendance,

    name = "Attendance",

    line = dict(color = '#17BECF'),

    opacity = 0.8)





data = [trace_high]



layout = dict(

    title='Yearly Attendance',

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label='1m',

                     step='month',

                     stepmode='backward'),

                dict(count=6,

                     label='6m',

                     step='month',

                     stepmode='backward'),

                dict(step='all')

            ])

        ),

        rangeslider=dict(

            visible = True

        ),

        type='date'

    )

)



fig = dict(data=data, layout=layout)

py.iplot(fig, filename = "Time Series with Rangeslider")
trace_high = go.Scatter(

    x=super_bowl.Date,

    y=super_bowl['Point Difference'],

    name = "Attendance",

    line = dict(color = '#17BECF'),

    opacity = 0.8)





data = [trace_high]



layout = dict(

    title='Yearly Point Difference',

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label='1m',

                     step='month',

                     stepmode='backward'),

                dict(count=6,

                     label='6m',

                     step='month',

                     stepmode='backward'),

                dict(step='all')

            ])

        ),

        rangeslider=dict(

            visible = True

        ),

        type='date'

    )

)



fig = dict(data=data, layout=layout)

py.iplot(fig, filename = "Time Series with Rangeslider")
winner_count = super_bowl['Winner'].value_counts()
loser_count = super_bowl['Loser'].value_counts()
trace1 = go.Bar(

    x=winner_count.index,

    y=winner_count.values,

    name='Won'

)

trace2 = go.Bar(

    x=loser_count.index,

    y=loser_count.values,

    name='Lost'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title= 'Wins vs Loss of teams'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
winner_count = super_bowl['QB  Winner'].value_counts()
loser_count = super_bowl['QB Loser'].value_counts()
trace1 = go.Bar(

    x=winner_count.index,

    y=winner_count.values,

    name='Won'

)

trace2 = go.Bar(

    x=loser_count.index,

    y=loser_count.values,

    name='Lost'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title= 'Wins vs Loss of Quaterbacks'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
winner_count = super_bowl['Coach Winner'].value_counts()
loser_count = super_bowl['Coach Loser'].value_counts()
trace1 = go.Bar(

    x=winner_count.index,

    y=winner_count.values,

    name='Won'

)

trace2 = go.Bar(

    x=loser_count.index,

    y=loser_count.values,

    name='Lost'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title= 'Wins vs Loss of Coaches'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
mvp_count = super_bowl['MVP'].value_counts()
import random

def random_colors(number_of_colors):

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

                 for i in range(number_of_colors)]

    return color
trace0 = go.Bar(

    x= mvp_count.index,

    y= mvp_count.values,

    marker=dict(

        color= random_colors(50),

       

        

    ),

    opacity=0.6

)



data = [trace0]

layout = go.Layout(

    title='Most MVP award in superbowl',

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='text-hover-bar')
state_codes = {

    'District of Columbia' : 'dc','Mississippi': 'MS', 'Oklahoma': 'OK', 

    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 

    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 

    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 

    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 

    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 

    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 

    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',

    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 

    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 

    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 

    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 

    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 

    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 

    'Nevada': 'NV', 'Maine': 'ME'}
super_bowl['state_code'] = super_bowl['State'].apply(lambda x : state_codes[x])
state_count = super_bowl['State'].value_counts()
data = [ dict(

        type='choropleth',

        colorscale = 'RdBu',

        autocolorscale = False,

        locations = super_bowl['state_code'],

        z = state_count.values,

        locationmode = 'USA-states',

        text = super_bowl['State'],

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "Hosting number")

        ) ]



layout = dict(

        title = 'Super Bowl hosting <br>(Hover for number)',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )

    

fig = dict( data=data, layout=layout )

py.iplot( fig, filename='d3-cloropleth-map' )
lucky= super_bowl.groupby(['Stadium'])
data = [dict(

  type = 'scatter',

  x = super_bowl['Winner'],

  y = super_bowl['Stadium'],

  mode = 'markers',

  transforms = [dict(

    type = 'groupby',

    groups = super_bowl['Stadium'],

   

  )]

)]



py.iplot({'data': data}, validate=False)
data = [dict(

  type = 'scatter',

  x = super_bowl['Referee'],

  y = super_bowl['Winner'],

  mode = 'markers',

  transforms = [dict(

    type = 'groupby',

    groups = super_bowl['Winner'],

   

  )]

)]



py.iplot({'data': data}, validate=False)
from PIL import Image



d = np.array(Image.open('../input/superbowl-pictures/s-l300.jpg'))
SB_DA = ' '.join(super_bowl['Winner'].tolist())
SB_DAA = "".join(str(v) for v in SB_DA).lower()
import matplotlib.pyplot as plt

from wordcloud import WordCloud

sns.set(rc={'figure.figsize':(11.7,8.27)})



wordcloud = WordCloud(mask=d,background_color="white").generate(SB_DAA)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.margins(x=0, y=0)

plt.title('Winning Teams',size=24)

plt.show()
team_stats = pd.read_csv('../input/team-sta/team_stats.csv' )

team_stats.head()
team_stats= team_stats.fillna(0)
df = team_stats.T

team_stats.info()
df = pd.melt(team_stats, id_vars=["Teams"])
df.head()
trace = go.Table(

    header=dict(values=list(df.columns),

                fill = dict(color='#C2D4FF'),

                align = ['left'] * 5),

    cells=dict(values=[df.Teams, df.variable, df.value],

               fill = dict(color='#F5F8FF'),

               align = ['left'] * 5))



data = [trace] 

py.iplot(data, filename = 'pandas_table')