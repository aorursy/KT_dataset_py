# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/ekosistem-javascript-di-indonesia/Ekosistem JavaScript (Responses) - Form Responses 1.csv")
df.head()
df.isnull().sum()
df.dtypes
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords,

        max_words=200,

        colormap='Set3',

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



show_wordcloud(df['Kamu sedang tertarik belajar tentang framework apa?'])
cnt_srs = df['Kegiatan saat ini'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Purples',

        reversescale = True

    ),

)



layout = dict(

    title='Kegiatan saat ini distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Kegiatan saat ini")
cnt_srs = df['Domisili saat ini (kota, provinsi)'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Reds',

        reversescale = True

    ),

)



layout = dict(

    title='Domisili saat ini distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="")

cnt_srs = df['Adakah komunitas JavaScript di sekitar tempat tinggal kamu? Jika ada, boleh disebutkan nama komunitasnya apa saja. Boleh meetup, telegram grup, WhatsApp, FB group dan sebagainya'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Greens',

        reversescale = True

    ),

)



layout = dict(

    title='Adakah komunitas JavaScript distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="")

def nonan(x):

    if type(x) == str:

        return x.replace("\n", "")

    else:

        return ""



text = ' '.join([nonan(abstract) for abstract in df["Saat ini menggunakan framework apa?"]])

wordcloud = WordCloud(max_font_size=None,colormap='Set3', background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

fig = px.imshow(wordcloud)

fig.update_layout(title_text='Framework')
def nonan(x):

    if type(x) == str:

        return x.replace("\n", "")

    else:

        return ""



text = ' '.join([nonan(abstract) for abstract in df["Bagaimana pendapatmu tentang framework tersebut? (Kelebihan dan kekurangan)"]])

wordcloud = WordCloud(max_font_size=None,colormap='Set3', background_color='green', collocations=False,

                      width=1200, height=1000).generate(text)

fig = px.imshow(wordcloud)

fig.update_layout(title_text='Framework')