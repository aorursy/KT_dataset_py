# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

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
df = pd.read_csv('../input/arabic-100k-reviews/ar_reviews_100k.tsv', sep='\t', header=None)
df.head().style.background_gradient(cmap='PRGn')
df = pd.DataFrame({'col':[0,1]})



df1 = pd.get_dummies(df['col']).add_prefix('topic')

print (df1)
sns.countplot(df1["topic0"])

plt.xticks(rotation=90)

plt.show()
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



show_wordcloud(df1['topic1'])
cnt_srs = df1['topic1'].value_counts().head()

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

    title='Text distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="topic1")
plt.figure(figsize=(10,8))

ax=sns.countplot(df1['topic1'])

ax.set_xlabel(xlabel="topic1",fontsize=17)

ax.set_ylabel(ylabel='No. of topics',fontsize=17)

ax.axes.set_title('Genuine No. of topics',fontsize=17)

ax.tick_params(labelsize=13)
from sklearn.model_selection import cross_val_score

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer



all_text=df1['topic1']

train_text=df1['topic1']

y=df1['topic0']


#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTmKgeAkrr6m4UyF5oAwQecnSF2IJX0AbbpQaWsBKVaO_LV37ei',width=400,height=400)