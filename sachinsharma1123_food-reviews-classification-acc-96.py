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
df=pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')
df
df.info()
import seaborn as sns

sns.heatmap(df.isnull())
df.isnull().sum()
sns.countplot(df['Score'])
#lets deop the unnecessary colmns from the datasets

df=df.drop(['Time','Summary','Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator'],axis=1)
df
#for convinience we will create two classes i.e 1 for score >=3 and 0 for score <3
df['Score']=np.where(df['Score']>=3,1,0)
df
sns.countplot(df['Score'])
#lets perform some text cleaning 

import nltk

from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

import re

import string

def text_cleaning(text):

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
df['Text']=df['Text'].apply(text_cleaning)
df
y=df['Score']

x=df['Text']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(ngram_range=(1,2),stop_words='english')

lr=LogisticRegression(max_iter=100000)

x_train=cv.fit_transform(x_train)



from sklearn.metrics import accuracy_score

lr.fit(x_train,y_train)

pred_1=lr.predict(cv.transform(x_test))

score_1=accuracy_score(y_test,pred_1)
score_1
from sklearn.naive_bayes import MultinomialNB

nb=MultinomialNB()

nb.fit(x_train,y_train)

pred_2=nb.predict(cv.transform(x_test))

score_2=accuracy_score(y_test,pred_2)
score_2