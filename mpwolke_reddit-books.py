#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
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



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/251403-bookthemed-reddit-entries-with-sentiment/Books_with_VAD.xlsx')



df.head()
df.dtypes
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
df.dropna(how = 'all',inplace = True)

df.drop(['text','title'],axis=1,inplace = True)

df.shape
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
df.corr()

plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=True,cmap='Blues')

plt.show()
sns.distplot(df["arousal"])
print ("Skew is:", df.dominance.skew())

plt.hist(df.dominance, color='pink')

plt.show()
sns.distplot(df["valence"].apply(lambda x: x**4))

plt.show()
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



show_wordcloud(df['redditor'])
fig = px.pie( values=df.groupby(['redditor']).size().values,names=df.groupby(['redditor']).size().index)

fig.update_layout(

    title = "Redditor",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)
cnt_srs = df['subreddit'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Blues',

        reversescale = True

    ),

)



layout = dict(

    title='',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="subreddit")
cnt_srs = df['redditor'].value_counts().head()

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

    title='',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="redditor")
fig = px.histogram(df[df.redditor.notna()],x="redditor",marginal="box",nbins=10)

fig.update_layout(

    title = "Redditor",

    xaxis_title="redditor",

    yaxis_title="score",

    barmode="group",

    bargap=0.1,

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0,

        dtick = 10),

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )

py.iplot(fig)
fig = px.histogram(df[df.dominance.notna()],x="dominance",marginal="box",nbins=10)

fig.update_layout(

    title = "Dominance",

    xaxis_title="dominance",

    yaxis_title="score",

    barmode="group",

    bargap=0.1,

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0,

        dtick = 10),

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#8b8b8b"

    )

    )

py.iplot(fig)
cat = []

num = []

for col in df.columns:

    if df[col].dtype=='O':

        cat.append(col)

    else:

        num.append(col)  

        

        

num
plt.style.use('dark_background')

for col in df[num].drop(['score'],axis=1):

    plt.figure(figsize=(8,5))

    plt.plot(df[col].value_counts(),color='Red')

    plt.xlabel(col)

    plt.ylabel('score')

    plt.tight_layout()

    plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRU7x4nV2xLea17b48x3Ipne76xY6tA_ubgo9M21XSd3eVuzTz6',width=400,height=400)