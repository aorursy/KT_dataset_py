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



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/freebase/topic.tsv', delimiter='\t',nrows=20,encoding='utf-8-sig')

df.head()
df.dtypes
df.isnull().sum()
df = df.dropna(subset=['alias', 'image', 'webpage', 'subjects', 'subject_of', 'properties', 'weblink', 'description'])
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



show_wordcloud(df['image'])
cnt_srs = df['image'].value_counts().head()

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

    title='Images distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Images")
df['name_length']=df['name'].apply(len)
sns.set(font_scale=2.0)



g = sns.FacetGrid(df,col='id',height=5)

g.map(plt.hist,'name_length')
plt.figure(figsize=(10,8))

ax=sns.countplot(df['name'])

ax.set_xlabel(xlabel="Names",fontsize=17)

ax.set_ylabel(ylabel='No. of names',fontsize=17)

ax.axes.set_title('Genuine No. of names',fontsize=17)

ax.tick_params(labelsize=13)
sns.set(font_scale=1.4)

plt.figure(figsize = (10,5))

sns.heatmap(df.corr(),cmap='coolwarm',annot=True,linewidths=.5)