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
df = pd.read_csv('../input/reviews/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
df
df.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot(df['Liked'])
#both the classes are equally distributed
#now clean the text for the modelling ,which is the important part
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
df['Review']=df['Review'].apply(text_cleaning)
df
y=df['Liked']

x=df['Review']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

cv=CountVectorizer(ngram_range=(1,2),stop_words='english')
x_train_trans=cv.fit_transform(x_train)

model_1=LogisticRegression(max_iter=10000)

model_1.fit(x_train_trans,y_train)

pred_1=model_1.predict(cv.transform(x_test))

score_1=accuracy_score(y_test,pred_1)
score_1
from sklearn.naive_bayes import MultinomialNB
model_2=MultinomialNB()

model_2.fit(x_train_trans,y_train)

pred_2=model_2.predict(cv.transform(x_test))

score_2=accuracy_score(y_test,pred_2)
score_2
from sklearn.svm import SVC

model_3=SVC()

model_3.fit(x_train_trans,y_train)

pred_3=model_3.predict(cv.transform(x_test))

score_3=accuracy_score(y_test,pred_3)
score_3