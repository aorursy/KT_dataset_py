#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
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
df = pd.read_excel('/kaggle/input//pune-bus-route-details-bus-stop-latlong/BRT  Non BRT Route Details  Bus Stop LatLong (1).xls')

df.head()
df = df.rename(columns={'Unnamed: 0':'unnamed', 'Unnamed: 1': 'unnamed1'})
df.isnull().sum()
df = df.dropna(subset=['unnamed', 'unnamed1'])
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



show_wordcloud(df['unnamed1'])
cnt_srs = df['unnamed1'].value_counts().head()

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

    title='Pune Bus Route ',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Routes")
df['unnamed1_length']=df['unnamed1'].apply(len)
sns.set(font_scale=2.0)



g = sns.FacetGrid(df,col='unnamed',height=5)

g.map(plt.hist,'unnamed1_length')
plt.figure(figsize=(10,8))

ax=sns.countplot(df['unnamed1'])

ax.set_xlabel(xlabel="Bus Routes",fontsize=17)

ax.set_ylabel(ylabel='No. of Routes',fontsize=17)

ax.axes.set_title('Genuine No. of Routes',fontsize=17)

ax.tick_params(labelsize=13)
sns.set(font_scale=1.4)

plt.figure(figsize = (10,5))

sns.heatmap(df.corr(),cmap='coolwarm',annot=True,linewidths=.5)
from sklearn.model_selection import cross_val_score

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer



all_text=df['unnamed1']

train_text=df['unnamed1']

y=df['unnamed']
word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 1),

    max_features=10000)

word_vectorizer.fit(all_text)

train_word_features = word_vectorizer.transform(train_text)
char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    stop_words='english',

    ngram_range=(2, 6),

    max_features=50000)

char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(train_text)



train_features = hstack([train_char_features, train_word_features])
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQRQjRmptlWEMYzTvpo69KrsOzCI6DzBHwLA5jgXYDsSg00M14m',width=400,height=400)