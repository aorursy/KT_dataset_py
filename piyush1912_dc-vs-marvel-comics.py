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
dc = pd.read_csv('../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv')

dc.head()
marvel = pd.read_csv('../input/fivethirtyeight-comic-characters-dataset/marvel-wikia-data.csv')

marvel.head()
sex_count = dc['SEX'].value_counts()

sex1_count = marvel['SEX'].value_counts()

trace1 = go.Bar(

    x=sex_count.index,

    y=sex_count.values,

    name='DC'

)

trace2 = go.Bar(

    x=sex1_count.index,

    y=sex1_count.values,

    name='Marvel'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title= 'Gender Comparisions in between DC and Marvel'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
sex_count = dc['ID'].value_counts()

sex1_count = marvel['ID'].value_counts()

trace1 = go.Bar(

    x=sex_count.index,

    y=sex_count.values,

    name='DC'

)

trace2 = go.Bar(

    x=sex1_count.index,

    y=sex1_count.values,

    name='Marvel'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title= 'Identity comparisions in between DC and Marvel'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
sex_count = dc['ALIGN'].value_counts()

sex1_count = marvel['ALIGN'].value_counts()

trace1 = go.Bar(

    x=sex_count.index,

    y=sex_count.values,

    name='DC'

)

trace2 = go.Bar(

    x=sex1_count.index,

    y=sex1_count.values,

    name='Marvel'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title= 'How many good and bad characters in between DC and Marvel?'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
sex_count = dc['ALIVE'].value_counts()

sex1_count = marvel['ALIVE'].value_counts()

trace1 = go.Bar(

    x=sex_count.index,

    y=sex_count.values,

    name='DC'

)

trace2 = go.Bar(

    x=sex1_count.index,

    y=sex1_count.values,

    name='Marvel'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title= 'Alive or Dead ?'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
trace0 = go.Bar(

    x= dc.YEAR,

    y= dc.APPEARANCES,

    name='DC Characters',

    text= dc.name,

    marker=dict(

        color='rgb(49,130,189)'

    )

)

trace1 = go.Bar(

    x= marvel.Year,

    y= marvel.APPEARANCES,

    name='Marvel Characters',

    text= marvel.name,

    marker=dict(

        color='rgb(204,204,204)',

    )

)



data = [trace0, trace1]

layout = go.Layout(

    xaxis=dict(tickangle=-45,

              title='Year'),

    yaxis=dict(title='Appearances'),

    title='Appearances with respect to Origin year Bar Plot',

    barmode='group',

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='angled-text-bar')
trace_high = go.Scatter(

                x=marvel.Year,

                y=marvel.APPEARANCES,

                name = "Marvel Appearances",

                line = dict(color = '#17BECF'),

                opacity = 0.8)



trace_low = go.Scatter(

                x=dc.YEAR,

                y=dc.APPEARANCES,

                name = "DC Appearances",

                line = dict(color = '#7F7F7F'),

                opacity = 0.8)



data = [trace_high,trace_low]



layout = dict(

    title='Appearances with respect to Origin year',

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label='1Y',

                     step='year',

                     stepmode='backward'),

                dict(count=6,

                     label='6Y',

                     step='year',

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
dc_top = dc.iloc[dc.groupby(dc['ALIGN'])['APPEARANCES'].idxmax()][['name', 'ALIGN']]
dc_top
dc_alive = dc.iloc[dc.groupby(dc['ALIVE'])['APPEARANCES'].idxmax()][['name', 'ALIVE']]
dc_alive
marvel_top = marvel.iloc[marvel.groupby(marvel['ALIGN'])['APPEARANCES'].idxmax()][['name', 'ALIGN']]
marvel_top
marvel_alive = marvel.iloc[marvel.groupby(marvel['ALIVE'])['APPEARANCES'].idxmax()][['name', 'ALIVE']]
marvel_alive
dc['name'] = dc['name'].replace({'Earth':' ', 'earth':' '})
from PIL import Image



d = np.array(Image.open('../input/comic-pict/images (17).jpeg'))
DC_DA = ' '.join(dc['name'].tolist())
DC_DAA = "".join(str(v) for v in DC_DA).lower()
import matplotlib.pyplot as plt

from wordcloud import WordCloud

sns.set(rc={'figure.figsize':(11.7,8.27)})



wordcloud = WordCloud(mask=d,background_color="white").generate(DC_DAA)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.margins(x=0, y=0)

plt.title('Popular Names of DC',size=24)

plt.show()
from PIL import Image



m = np.array(Image.open('../input/comic-pict/images (18).jpeg'))
M_DA = ' '.join(marvel['name'].tolist())
M_DAA = "".join(str(v) for v in M_DA).lower()
import matplotlib.pyplot as plt

from wordcloud import WordCloud

sns.set(rc={'figure.figsize':(11.7,8.27)})



wordcloud = WordCloud(mask=m,background_color="white").generate(M_DAA)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.margins(x=0, y=0)

plt.title('Popular Names of Marvel',size=24)

plt.show()
dc['comics']= 'DC'
dc= dc.truncate(before=-1, after=20)
import networkx as nx

FG = nx.from_pandas_edgelist(dc, source='comics', target='name', edge_attr=True,)
nx.draw_networkx(FG, with_labels=True)
marvel['comics'] = 'Marvel'
marvel = marvel.truncate(before=-1, after=20)
import networkx as nx

FG1 = nx.from_pandas_edgelist(marvel, source='comics', target='name', edge_attr=True,)
nx.draw_networkx(FG1, with_labels=True)