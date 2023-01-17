# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import plotly.offline as py
from plotly.offline import init_notebook_mode,iplot
from plotly import tools
import plotly.graph_objs as go
init_notebook_mode(connected=True)
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize']=(15,12)

import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv('/kaggle/input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv')
data.head()
#view data info
data.info()
#Describe some statistics
data.describe()
data.isnull().any().any()
data.isnull().sum()
data.columns
data['Review Text'].isna().sum()
data['Review Text'] = data[~data['Review Text'].isna()]

data.isnull().sum()
ratings=data['Rating'].value_counts()

label_rating=ratings.index
size_rating=ratings.values

color=['pink','yellow','darkgrid','aqua','gold']

rating_piegraph=go.Pie(labels=label_rating,
                      values=size_rating,
                      marker=dict(colors=color),
                      name='Clothing',hole=0.3)

df=[rating_piegraph]

layout=go.Layout(title="Distribution of Women's E-Commerce Clothing Reviews")

fig=go.Figure(data=df,
             layout=layout)

py.iplot(fig)

data.dropna(inplace=True)
data.isnull().sum()

color=plt.cm.copper(np.linspace(0,1,15))
data['Division Name'].value_counts().plot.bar(color=color,figsize=(15,12))
plt.title("Distribution of E-commerce Clothing Reviews")
plt.xlabel('Division Name')
plt.ylabel('Count')
plt.show()
data['Rating'].value_counts().plot.hist(color='skyblue',figsize=(15,12))
plt.title('Distribution of E-commerce Clothing Reviews Rating')
plt.xlabel('Rating')
plt.ylabel('count')
plt.show()
data['Department Name'].value_counts().plot.bar(color='green',figsize=(15,12))
plt.title('E-Commerce Clothing Department Name')
plt.xlabel('Department Name')
plt.ylabel('Count')
plt.show()
color=plt.cm.ocean(np.linspace(0,1,20))
sns.boxenplot(data['Department Name'],data['Rating'],palette='spring')
plt.title('Department Name vs Rating')
plt.xticks(rotation=90)
plt.show()
sns.swarmplot(data['Department Name'],data['Rating'],palette='cool')
plt.title('Department Name vs Rating')
plt.xticks(rotation=90)
plt.show()
sns.violinplot(data['Recommended IND'],data['Rating'],palette='deep')
plt.title('Recommended IND wise Rating')
plt.show()
sns.boxplot(data['Rating'],data['Recommended IND'],palette='Blues')
plt.title('Rating vs Recommended IND')
plt.show()
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from nltk.stem.porter import PorterStemmer
import string
punct=string.punctuation
punct
from spacy.lang.en import English
parser=English()
data.head()
nlp=spacy.load('en_core_web_sm')
corpus=[]
for i in range(0,3150):
   
    review=re.sub('[^a-z A-Z]','',data['Review Text'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if word not in STOP_WORDS and word not in punct]
    review=' '.join(review)
    corpus.append(review)

data['Review']=data['Review Text']
data.head()
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
cv=CountVectorizer(STOP_WORDS)
words=cv.fit_transform(data['Review Text'])
words_sum=words.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq,key=lambda x:x[1],reverse=True)

frequency=pd.DataFrame(words_freq,columns=['words','freq'])
color=plt.cm.ocean(np.linspace(0,1,20))
frequency.head(20).plot.bar(X='word',y='freq',color=color,figsize=(15,12))
plt.title('Most frequently occurning Top-20 words')
plt.show()
import spacy
from spacy import displacy

for i in range(15,50):
    one_sentence=data['Review Text'][i]
    doc=nlp(one_sentence)
    print(displacy.render(doc,style='ent',jupyter=True))
