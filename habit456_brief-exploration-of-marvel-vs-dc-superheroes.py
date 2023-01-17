import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected=True)  

import plotly.figure_factory as ff

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
def col_trim(df, *cols):

    """Takes a DataFrame, df, and any number of its columns, *cols, as args and trims the last word of each value.

    To avoid repitition in words. For example: Blue Hair, Brown Hair, Bald becomes Blue, Brown, Bald"""

    def f(x):

        y = str(x)

        y = y.split()

        if len(y) > 1:

            return " ".join(str(x).split()[:-1])

        return x

    for col in cols:

        df[col] = df[col].map(f)
dc = pd.read_csv('../input/dc-wikia-data.csv', index_col='name')

dc = dc.drop(['GSM', 'urlslug', 'page_id'], 1)

col_trim(dc, 'EYE', 'HAIR', 'SEX', 'ID', 'ALIGN', 'ALIVE')

dc.head()
marvel = pd.read_csv('../input/marvel-wikia-data.csv', index_col="name")

marvel = marvel.drop(['GSM', 'urlslug', 'page_id'], 1)

col_trim(marvel, 'EYE', 'HAIR', 'SEX', 'ID', 'ALIGN', 'ALIVE')

marvel.head()
df_whole = marvel.append(dc, sort=True)
eye_color_count_m = marvel['EYE'].value_counts()

eye_color_count_d = dc['EYE'].value_counts()



trace1 = go.Bar(

    x=eye_color_count_m.index,

    y=eye_color_count_m.values,

    name='Marvel'

)

trace2 = go.Bar(

    x=eye_color_count_d.index,

    y=eye_color_count_d.values,

    name='DC'

)

data = [trace1, trace2]



layout = go.Layout(

    barmode='group',

    title='Eye Color Comparisons Between DC and Marvel'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
hair_count_m = marvel['HAIR'].value_counts()

hair_count_d = dc['HAIR'].value_counts()



trace1 = go.Bar(

    x=hair_count_m.index,

    y=hair_count_m.values,

    name='Marvel'

)

trace2 = go.Bar(

    x=hair_count_d.index,

    y=hair_count_d.values,

    name='DC'

)

data = [trace1, trace2]



layout = go.Layout(

    barmode='group',

    title='Hair Color Comparisons Between DC and Marvel'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
vc = df_whole.ALIGN.value_counts()



colors = ['pink', 'aqua', 'gold', 'lightblue']



trace = go.Pie(

    labels  = vc.index,

    values  = vc.values,

    name    = 'Alignment',

    hole    = 0.3,

    marker  = dict(colors=colors)

)



data = [trace]



layout = go.Layout(

    title="Characters By Alignment"

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)