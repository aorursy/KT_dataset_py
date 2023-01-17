# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn

%matplotlib inline

%pylab inline

pylab.rcParams['figure.figsize'] = (20, 11)

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/nba-all-star-game-20002016/NBA All Stars 2000-2016 - Sheet1.csv')
df.head()
df = df.set_index('Year')
df.head()
plt.title("WT & Time", fontsize=16)

df["WT"].plot(grid=True);
labels = df['Selection Type'].value_counts().index

values = df['Selection Type'].value_counts().values



fig = px.pie(df, values=values, names=labels, title='Selection Type Distribution')

fig.show()
labels = df['Team'].value_counts().index

values = df['Team'].value_counts().values



fig = px.pie(df, values=values, names=labels, title='All-star players by team')

fig.show()
labels = df['Nationality'].value_counts().index

values = df['Nationality'].value_counts().values



fig = px.pie(df, values=values, names=labels, title='Nationality Distribution')

fig.show()
labels = df['HT'].value_counts().index

values = df['HT'].value_counts().values



fig = px.pie(df, values=values, names=labels, title='HT Distribution')

fig.show()
labels = df['Pos'].value_counts().index

values = df['Pos'].value_counts().values



fig = px.pie(df, values=values, names=labels, title='Position Distribution')

fig.show()
def plot_pie_charts(dataframe, features):

    for feature in features:

        labels = dataframe[feature].value_counts().index

        values = dataframe[feature].value_counts().values

        fig = px.pie(df, values=values, names=labels, title=f'{feature} Distribution')

        fig.show()
features = ['Pos', 'Nationality', 'HT', 'Selection Type', 'Team']

plot_pie_charts(df, features)
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, max_words=None, max_font_size=None, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=max_words if max_words else 200,

        max_font_size=max_font_size if max_font_size else 36, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

    ).generate(str(data))



    fig = plt.figure(1, figsize=(40, 20))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(df['Player'].values, max_words=150, max_font_size=48)