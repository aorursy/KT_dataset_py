#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRnRXjirMDmVHwnwh1389zK_rrupRo5IrFPg9WjV7wkR6iDBj1c',width=400,height=400)
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
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSXyDsHqJD42qZgVZITJbUrwYpJZCVxMJbMhcf5xpmW8QfFm-w0',width=400,height=400)
df = pd.read_excel('/kaggle/input/climate-change/Climate_Change.xls')

df.head()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQuVKdyC-ZPBb0SMhSOy77kFiweG79UsWtV_FBcbQXn7M2G8Oss',width=400,height=400)
df.dtypes
df = df.rename(columns={'Country name':'country', 'Country code': 'code', 'Series code': 'series', 'Series name': 'name'})
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQSXuPza7ovtGcwYpsfcqbgDly0Jj9Tn4t-pv01f-ua_6qMediQ',width=400,height=400)
df.isnull().sum()
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

    

show_wordcloud(df['country'])
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSW70JdLjpHWqvKSkn1Q8WygHgFw9j_9YfOhdOHZTThQGIU4sOH',width=400,height=400)
cnt_srs = df['code'].value_counts().head()

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

    title='Codes distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="code")
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRVJrb7Z5ApgRUuBIs5u459umh2Oyyhkba1uw5r6NpbTHpBmDZ9',width=400,height=400)
df['code_length']=df['code'].apply(len)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS-E5w5bGVhBRwmU8f6QzF0YHmr5-K9f910w3JhxSik4P7KHBPD',width=400,height=400)
sns.set(font_scale=2.0)



g = sns.FacetGrid(df,col='series',height=5)

g.map(plt.hist,'code_length')
plt.figure(figsize=(10,8))

ax=sns.countplot(df['Decimals'])

ax.set_xlabel(xlabel="Decimals",fontsize=17)

ax.set_ylabel(ylabel='No. of Decimals',fontsize=17)

ax.axes.set_title('Genuine No. of Decimals',fontsize=17)

ax.tick_params(labelsize=13)
from sklearn.model_selection import cross_val_score

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer



all_text=df['series']

train_text=df['series']

y=df['name']
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
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTXEEyGGsnh_jlt78ekUdvDq3UwNTqa0tjBDNMW92WMrm-1LBFL',width=400,height=400)
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

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSfItz29a-KLPRaAWkjlxBIld_pjbzlKornet54oEfIJ7Y0w5JG',width=400,height=400)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_features, y,test_size=0.3,random_state=101)
import xgboost as xgb
xgb=xgb.XGBClassifier()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSOTf13-wqEUWJAuvEhIbA9DcvsPNmp0tz-hw2hOSpOZ_QlRLC2',width=400,height=400)
#xgb.fit(X_train,y_train)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSIyXSuk6SqjzMesXuygr_f-FRz7a_xHgTz9KWZIxb2qxFqDD-X',width=400,height=400)