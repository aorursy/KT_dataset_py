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
df=pd.read_csv('/kaggle/input/fake-news-detection/data.csv')
df.head()
for i in range(0,df.shape[0]-1):

    if (df.Body.isnull()[i]):

        df.Body[i]=df.Headline[i]
df.isnull().sum()
import re 

import nltk

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk import pos_tag

from nltk.corpus import stopwords

import string
print('Data Cleaning Started.....')



stop=set(stopwords.words('english'))

punc=list(string.punctuation)

stop.update(punc)



RE_EMOJI=re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)

def remove_emoji(text):

    return RE_EMOJI.sub(r' ',text)



RE_HTMLTAG=re.compile(r'<.*?>')

def remove_htmltag(text):

    return RE_HTMLTAG.sub(r' ',text)



RE_BLANKLINE=re.compile(r'^(?:[\t ]*(?:\r?\n|\r))+')

def remove_blankline(text):

    return RE_BLANKLINE.sub(r' ',text)



RE_EMAIL=re.compile(r'[\w._%+-]{1-20}@[\w.-]{2,20}.[A-Za-z]{2-3}')

def remove_email(text):

    return RE_EMAIL.sub(r' ',text)



def remove_stopwords(text):

    filtered=[]

    word_token=word_tokenize(text)

    for word in word_token:

        if word not in stop:

            filtered.append(word)

    text=' '.join(filtered)

    return text



def clean_data(df,columns:list):

    for col in columns:

        df[col]=df[col].apply(lambda x:remove_emoji(x))

        df[col]=df[col].apply(lambda x:remove_htmltag(x))

        df[col]=df[col].apply(lambda x:remove_blankline(x))

        df[col]=df[col].apply(lambda x:remove_email(x))

        df[col]=df[col].apply(lambda x:remove_stopwords(x))

    return df
col=['Body']

clean_data(df,col)

print('Data Cleaning Completed...')
from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

text = " ".join(i for i in df['Body'])



wordcloud = WordCloud(max_font_size = 50, 

                     background_color = "white").generate(text)

plt.figure(figsize = [10,10])

plt.imshow(wordcloud, interpolation = "bilinear")

plt.axis("off")

plt.show()
df=df.drop(['URLs'],axis=1)
x=df['Body']+df['Headline']

y=df.Label
from sklearn.model_selection import train_test_split as tts

from sklearn.feature_extraction.text import TfidfVectorizer

import itertools

from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

from sklearn.linear_model import LogisticRegression
train_x,test_x,train_y,test_y=tts(x,y,test_size=0.2,stratify=y)
tfidf_vect = TfidfVectorizer(stop_words = 'english')

tfidf_train = tfidf_vect.fit_transform(train_x)

tfidf_test = tfidf_vect.transform(test_x)

tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vect.get_feature_names())
tfidf_df
clf=MultinomialNB()

clf.fit(tfidf_train,train_y)

pred=clf.predict(tfidf_test)

score=metrics.accuracy_score(test_y,pred)

print('Score : ',score)
lr=LogisticRegression()

lr.fit(tfidf_train,train_y)

pred=lr.predict(tfidf_test)

train=lr.predict(tfidf_train)

score=metrics.accuracy_score(test_y,pred)

score1=metrics.accuracy_score(train_y,train)

print('Test Score : ',score)

print('Training Score :',score1)