# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib

import matplotlib.pyplot as plt # for plotting

import seaborn as sns # for making plots with seaborn

color = sns.color_palette()

# import plotly.offline as py

# py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()

from plotly import tools

import plotly.tools as tls

import squarify

from mpl_toolkits.basemap import Basemap

from numpy import array

from matplotlib import cm



import warnings

warnings.filterwarnings('ignore')
df1=pd.read_csv('../input/athlete_events.csv')

df1.head()
df1.shape
print(' Total of',df1['Sport'].nunique(),'unique sports were played. \n \n Following is the list:\n \n', df1['Sport'].unique())
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



show_wordcloud(df1['Sport'])
fig = {

  "data": [

    {

      "values": df1['Sex'].value_counts(),

      "labels": [

        "Male",

        "Female",

      ],

        'marker': {'colors': ['rgb(175, 49, 35)',

                                  'rgb(177, 180, 34)']},

      "name": "Sex Ratio of Participants",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

     "layout": {

        "title":"Sex Ratio of Participants"

     }

}

iplot(fig, filename='donut')
y0 = df1.ix[df1['Sex']=='M']['Age']

y1 = df1.ix[df1['Sex']=='F']['Age']



trace0 = go.Box(

    y=y0,

    name="Age Distribution for Male")

trace1 = go.Box(

    y=y1,

    name="Age Distribution for Female")

data = [trace0, trace1]

iplot(data)
df_medals=df1.ix[df1['Medal']=='Gold']



cnt_srs = df_medals['Team'].value_counts().head(20)



trace = go.Bar(

    x=cnt_srs.index,

    y=cnt_srs.values,

    marker=dict(

        color="blue",

        #colorscale = 'Blues',

        reversescale = True

    ),

)



layout = go.Layout(

    title='Top 20 countries with Maximum Gold Medals'

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="medal")  
cnt_srs = df1['Sport'].value_counts()



trace = go.Bar(

    x=cnt_srs.index,

    y=cnt_srs.values,

    marker=dict(

        color=cnt_srs.values,

        colorscale = 'Picnic',

        reversescale = True

    ),

)



layout = go.Layout(

    title='Most Popular Sport'

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="sport")
df_usa=df1.ix[(df1['Team']=='United States')]

df_usa_medal=df_usa.ix[df_usa['Medal']=='Gold']



medal_map = {'Gold':1}

df_usa_medal['Medal'] = df_usa_medal['Medal'].map(medal_map)



df_usa_sport=df_usa_medal.groupby(['Sport'],as_index=False)['Medal'].agg('sum')



df_usa_sport=df_usa_sport.sort_values(['Medal'],ascending=False)



df_usa_sport=df_usa_sport.head(10)



colors = ['#91BBF4', '#91F4F4', '#F79981', '#F7E781', '#C0F781','rgb(32,155,160)', 'rgb(253,93,124)', 'rgb(28,119,139)', 'rgb(182,231,235)', 'rgb(35,154,160)']



n_phase = len(df_usa_sport['Sport'])

plot_width = 200



# height of a section and difference between sections 

section_h = 100

section_d = 10



# multiplication factor to calculate the width of other sections

unit_width = plot_width / max(df_usa_sport['Medal'])



# width of each funnel section relative to the plot width

phase_w = [int(value * unit_width) for value in df_usa_sport['Medal']]



height = section_h * n_phase + section_d * (n_phase - 1)



# list containing all the plot shapes

shapes = []



# list containing the Y-axis location for each section's name and value text

label_y = []



for i in range(n_phase):

        if (i == n_phase-1):

                points = [phase_w[i] / 2, height, phase_w[i] / 2, height - section_h]

        else:

                points = [phase_w[i] / 2, height, phase_w[i+1] / 2, height - section_h]



        path = 'M {0} {1} L {2} {3} L -{2} {3} L -{0} {1} Z'.format(*points)



        shape = {

                'type': 'path',

                'path': path,

                'fillcolor': colors[i],

                'line': {

                    'width': 1,

                    'color': colors[i]

                }

        }

        shapes.append(shape)

        

        # Y-axis location for this section's details (text)

        label_y.append(height - (section_h / 2))



        height = height - (section_h + section_d)

        

label_trace = go.Scatter(

    x=[-200]*n_phase,

    y=label_y,

    mode='text',

    text=df_usa_sport['Sport'],

    textfont=dict(

        color='rgb(200,200,200)',

        size=15

    )

)

 

# For phase values

value_trace = go.Scatter(

    x=[-350]*n_phase,

    y=label_y,

    mode='text',

    text=df_usa_sport['Medal'],

    textfont=dict(

        color='rgb(200,200,200)',

        size=12

    )

)



data = [label_trace, value_trace]

 

layout = go.Layout(

    title="<b>Top 10 Sports in which USA is best</b>",

    titlefont=dict(

        size=12,

        color='rgb(203,203,203)'

    ),

    shapes=shapes,

    height=600,

    width=800,

    showlegend=False,

    paper_bgcolor='rgba(44,58,71,1)',

    plot_bgcolor='rgba(44,58,71,1)',

    xaxis=dict(

        showticklabels=False,

        zeroline=False,

    ),

    yaxis=dict(

        showticklabels=False,

        zeroline=False

    )

)

 

fig = go.Figure(data=data, layout=layout)

iplot(fig)
df_china=df1.ix[(df1['Team']=='China')]

df_china_medal=df_china.ix[df_china['Medal']=='Gold']



medal_map = {'Gold':1}

df_china_medal['Medal'] = df_china_medal['Medal'].map(medal_map)



df_china_sport=df_china_medal.groupby(['Sport'],as_index=False)['Medal'].agg('sum')



df_china_sport=df_china_sport.sort_values(['Medal'],ascending=False)



df_china_sport=df_china_sport.head(10)



temp_series = df_china_sport['Medal']

labels = df_china_sport['Sport']

sizes = (np.array((temp_series / temp_series.sum())*100))



trace = go.Pie(labels=labels, values=sizes)

layout = go.Layout(

    title='Sports in which China has won maximum Gold Medals',

    width=900,

    height=900,

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="china")
df_medal=df1.dropna(subset=['Medal'])



df_medal_male=df_medal.ix[df_medal['Sex']=="M"]

df_medal_female=df_medal.ix[df_medal['Sex']=="F"]
df_medal_male_gold=df_medal_male.ix[df_medal_male['Medal']=='Gold']



medal_map = {'Gold':1}

df_medal_male_gold['Medal'] = df_medal_male_gold['Medal'].map(medal_map)



df_medal_male_gold=df_medal_male_gold.groupby(['Sport'],as_index=False)['Medal'].agg('sum')



df_medal_female_gold=df_medal_female.ix[df_medal_female['Medal']=='Gold']



df_medal_female_gold['Medal'] = df_medal_female_gold['Medal'].map(medal_map)



df_medal_female_gold=df_medal_female_gold.groupby(['Sport'],as_index=False)['Medal'].agg('sum')
temp1 = df_medal_male_gold[['Sport', 'Medal']] 

temp2 = df_medal_female_gold[['Sport', 'Medal']] 

# temp1 = gun[['state', 'n_killed']].reset_index(drop=True).groupby('state').sum()

# temp2 = gun[['state', 'n_injured']].reset_index(drop=True).groupby('state').sum()

trace1 = go.Bar(

    x=temp1.Sport,

    y=temp1.Medal,

    name = 'Sports in which Males have won max. Gold Medals'

)

trace2 = go.Bar(

    x=temp2.Sport,

    y=temp2.Medal,

    name = 'Sports in which Females have won max. Gold Medals'

)



fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Sports in which Males have won max. Gold Medals', 'Sports in which Females have won max. Gold Medals'))

                                                          



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

                          

fig['layout']['xaxis1'].update(title='Name of Sport')

fig['layout']['xaxis2'].update(title='Name of Sport')



fig['layout']['yaxis1'].update(title='Sports in which Males have won Gold Medals')

fig['layout']['yaxis2'].update(title='Sports in which Females have won Gold Medals')

                          

fig['layout'].update(height=500, width=1500, title='Sports in which Males and Females have won max. Gold Medals')

iplot(fig, filename='simple-subplot')
df_india=df1.ix[df1['Team']=="India"]

df_india_medal=df_india.dropna(subset=['Medal'])
df_india_medal.head(1)
cnt_srs = df_india_medal['Sport'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1]

    ),

)



layout = dict(

    title='Sports at which India are good at:',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="india_sports")
df1.head(2)
df_medals = df1.dropna(subset=['Medal']).reset_index(drop=True)

df_medals.drop_duplicates(inplace=True)

df_medals = pd.DataFrame(df_medals.groupby(['Team', 'Medal'])['Medal'].agg({'Medal_Count':'count'})).reset_index()

df_medals.head()
df_medals = pd.merge(df_medals,pd.DataFrame(df_medals.groupby('Team')['Medal_Count'].sum()).reset_index().rename(columns={'Medal_Count':'Total_Medals'}), on=['Team'], how='left')

df_medals = df_medals.sort_values(by=['Total_Medals'], ascending=False)

df_medals.reset_index(drop=True, inplace=True)

df_medals = df_medals.head(30) # Top 10 Countries with max. medals

df_medals.head()
fig = go.Figure(data=[

    go.Bar(name='Bronze',x=df_medals[df_medals['Medal']=='Bronze']['Team'], y=df_medals[df_medals['Medal']=='Bronze']['Medal_Count']),

    go.Bar(name='Silver', x=df_medals[df_medals['Medal']=='Silver']['Team'], y=df_medals[df_medals['Medal']=='Silver']['Medal_Count']),

    go.Bar(name='Gold', x=df_medals[df_medals['Medal']=='Gold']['Team'], y=df_medals[df_medals['Medal']=='Gold']['Medal_Count'])

])

# Change the bar mode

fig.layout.update(barmode='stack', title='Top 10 Country with max. Medals')

iplot(fig)