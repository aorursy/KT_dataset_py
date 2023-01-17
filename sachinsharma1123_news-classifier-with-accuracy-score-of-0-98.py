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
df1=pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
df2=pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
df1
df2
#we will take 1 for true news and 0 for fake news
df1['Target']=1
df1
df2['Target']=0
df2
df=pd.concat([df1,df2],axis=0)
df
import seaborn as sns

sns.heatmap(df.isnull())
df
df=df.drop(['title','subject','date'],axis=1)
df
#now some text processing

import nltk

from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

import re

import string

def custom_preprocessor(text):

    '''

    Make text lowercase, remove text in square brackets,remove links,remove special characters

    and remove words containing numbers.

    '''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub("\\W"," ",text) # remove special chars

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    

    return text
df['text']=df['text'].apply(custom_preprocessor)
df
y=df['Target']

x=df['text']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer()

x_trn_vec = vectorizer.fit_transform(x_train)
model_1=LogisticRegression()

model_1.fit(x_trn_vec,y_train)
pred_1=model_1.predict(vectorizer.transform(x_test))
score_1=accuracy_score(y_test,pred_1)
score_1
final_df=pd.DataFrame({'actual':y_test,

                      'predicted':pred_1})
final_df