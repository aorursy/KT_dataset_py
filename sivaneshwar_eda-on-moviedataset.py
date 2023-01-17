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


import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import seaborn as sns



import json

import warnings

warnings.filterwarnings('ignore')

import base64

import io

from scipy.misc import imread

import codecs

from IPython.display import HTML
movies=pd.read_csv('../input/tmdb_5000_movies.csv')

mov=pd.read_csv('../input/tmdb_5000_credits.csv')
def from_json(df,attr):

    df[attr] = df[attr].apply(json.loads)

    

    for index,i in zip(df.index, df[attr]):

        l=[]

        for j in range(len(i)):

            l.append(i[j]['name'])

        df.loc[index,attr] = str(l)

    
json_columns = ['keywords','production_companies','production_countries','spoken_languages']
for c in json_columns:

    from_json(movies,c)
from_json(mov,'cast')


def director(x):

    for i in x:

        if i['job'] == 'Director':

            return i['name']

    else:

        return "Unknown"

mov['crew']=mov['crew'].apply(json.loads)

mov['crew']=mov['crew'].apply(director)

mov.rename(columns={'crew':'director'},inplace=True)
movies = movies.merge(mov, left_on='id', right_on='movie_id',how='left')
movies.columns
movies.head(1)
movies['PoL']=movies.revenue - movies.budget
df=movies[['id','original_title','genres','cast','vote_average','vote_count','popularity','director','keywords','runtime','budget','revenue','PoL']]

df.head(2)
import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



df = movies.sort_values(by='budget',ascending=False).iloc[:20,:]
df = df[df.vote_count>1000].iloc[:30,:]

df = df.sort_values(by='vote_average',ascending=False)
data = [

    {

        'y': df.vote_average,

        'x': df.original_title,

        'mode': 'markers',

        'marker': {

            'color': df.popularity,

            'size': df.vote_count//100,

            'showscale': True

        },

        "text" :df.PoL//1000000

    }

]

iplot(data)
from wordcloud import WordCloud
kw = movies['keywords'].values.tolist()
kw = [eval(x) for x in kw]
kw = sum(kw,[])
plt.subplots(figsize=(20,10))

wordcloud = WordCloud(

                          background_color='white',

                          width=720,

                          height=480

                         ).generate(" ".join(kw))

plt.imshow(wordcloud)

plt.axis('off')

#plt.savefig('graph.png')



plt.show()


movies.tagline.isnull().sum()


tg = movies['tagline'].dropna().values.tolist()



plt.subplots(figsize=(20,10))

wordcloud = WordCloud(

                          background_color='black',

                          width=720,

                          height=480

                         ).generate(" ".join(tg))

plt.imshow(wordcloud)

plt.axis('off')

#plt.savefig('graph.png')



plt.show()
dirpol = movies[['director','PoL']]
dirpol = dirpol.groupby(['director']).sum()

topdir = dirpol.sort_values(by='PoL',ascending=False).iloc[:20,:]
dirmovcnt = movies[['director','original_title']].groupby(['director']).count()

dirmovcnt.columns = ['Num_movies']
topdirmovcnt = dirmovcnt.sort_values(by='Num_movies',ascending=False)
type(topdir.index.tolist())
sizes = []

for name in topdir.index.tolist():

    sizes.append(topdirmovcnt[topdirmovcnt.index==name]['Num_movies'].values[0])
sizes
data = [

    {

        'y': topdir.PoL,

        'x': topdir.index,

        'mode': 'markers',

        'marker': {

            'color': 'red',

            'size': sizes,

            'showscale': False

        },

        "text" :sizes

    }

]

iplot(data)
for i in range(len(movies)):

    try:

        movies.loc[i,'Lead'] = eval(movies.cast[i])[0]

    except:

        movies.loc[i,'Lead']='Unknown'
leadpol = movies[['Lead','PoL']]

leadpol = leadpol.groupby(['Lead']).sum()



toplead = leadpol.sort_values(by='PoL',ascending=False).iloc[:20,:]

leadmovcnt = movies[['Lead','original_title']].groupby(['Lead']).count()





leadmovcnt.columns = ['Num_movies']



topleadmovcnt = leadmovcnt.sort_values(by='Num_movies',ascending=False)



sizes = []

for name in toplead.index.tolist():

    sizes.append(topleadmovcnt[topleadmovcnt.index==name]['Num_movies'].values[0])



data = [

    {

        'y': toplead.PoL,

        'x': toplead.index,

        'mode': 'markers',

        'marker': {

            'color': 'red',

            'size': sizes,

            'showscale': False

        },

        "text" :sizes

    }

]

iplot(data)
def comp_dirs(d1,d2):

    



    trace0 = go.Box(

        y=movies[movies.director==d1].PoL,

        name = d1,

        marker = dict(

            color = 'rgb(12, 12, 140)',

        )

    )

    trace1 = go.Box(

        y=movies[movies.director==d2].PoL,

        name = d2,

        marker = dict(

            color = 'rgb(12, 128, 128)',

        )

    )

    data = [trace0, trace1]

    layout = dict(

        title="Director Comparision P/L"

    )

    fig = dict(

        data = data,

        layout = layout

    )

    iplot(fig)

            
comp_dirs('James Cameron', 'Christopher Nolan')




comp_dirs('Sam Raimi', 'Christopher Nolan')
import random
D1 = random.choice(topdir.index.tolist())

D2 = random.choice(topdir.index.tolist())

comp_dirs(D1,D2)
def comp_leads_V(l1,l2):

    trace0 = go.Violin(

        y=movies[movies.Lead==l1].PoL,

        name = l1,

        marker = dict(

            color = 'rgb(12, 12, 140)',

        )

    )

    trace1 = go.Violin(

        y=movies[movies.Lead==l2].PoL,

        name = l2,

        marker = dict(

            color = 'rgb(12, 128, 128)',

        )

    )

    data = [trace0, trace1]

    layout = dict(

        title="Lead Actror Comparision P/L"

    )

    fig = dict(

        data = data,

        layout = layout

    )

    iplot(fig)

            
def comp_leads_B(l1,l2):

    trace0 = go.Box(

        y=movies[movies.Lead==l1].PoL,

        name = l1,

        marker = dict(

            color = 'rgb(12, 12, 140)',

        )

    )

    trace1 = go.Box(

        y=movies[movies.Lead==l2].PoL,

        name = l2,

        marker = dict(

            color = 'rgb(12, 128, 128)',

        )

    )

    data = [trace0, trace1]

    layout = dict(

        title="Lead Actror Comparision P/L"

    )

    fig = dict(

        data = data,

        layout = layout

    )

    iplot(fig)

            
L1 = random.choice(toplead.index.tolist())

L2 = random.choice(toplead.index.tolist())

comp_leads_V(L1,L2)
L1 = random.choice(toplead.index.tolist())

L2 = random.choice(toplead.index.tolist())

comp_leads_B(L1,L2)