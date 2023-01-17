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
df=pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')
df
df.isnull().sum()
#drop the unnecessary features from the dataset
df=df.drop(['reviewerID','asin','reviewerName','unixReviewTime','reviewTime','summary'],axis=1)
df=df.drop('helpful',axis=1)
df.isnull().sum()
df=df.dropna()
df
#lets create a binary rating i.e 1 if overall rating is greater than equal to 3 and 0 if less than 3
df['overall']=np.where(df['overall']>=3,1,0)
df
import seaborn as sns

sns.countplot(x=df['overall'],data=df)
df['overall'].value_counts()
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
df['reviewText']=df['reviewText'].apply(text_cleaning)
x=df['reviewText']

y=df['overall']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(ngram_range=(1,3))

x_train_trans=cv.fit_transform(x_train)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr=LogisticRegression(max_iter=10000)
lr.fit(x_train_trans,y_train)

pred_1=lr.predict(cv.transform(x_test))

score_1=accuracy_score(y_test,pred_1)
score_1
from sklearn.svm import SVC

svm=SVC()
svm.fit(x_train_trans,y_train)

pred_2=svm.predict(cv.transform(x_test))

score_2=accuracy_score(y_test,pred_2)
score_2
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(x_train_trans,y_train)

pred_3=clf.predict(cv.transform(x_test))

score_3=accuracy_score(y_test,pred_3)
score_3