# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn import linear_model

from sklearn import metrics

from sklearn.model_selection import train_test_split

%matplotlib inline 

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from numpy import array

from numpy import argmax

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn import svm

from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer

from sklearn.feature_selection import chi2

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers.core import Dense, Activation

from keras.utils import np_utils

import re

import seaborn as sns

from keras.preprocessing import sequence

from keras.preprocessing.text import one_hot

from keras.preprocessing.text import text_to_word_sequence

from sklearn.svm import LinearSVC
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv',delimiter=',',encoding='latin-1', na_filter=False)

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv',delimiter=',',encoding='latin-1', na_filter=False)

train.head(10)
train.sentiment.value_counts().plot(figsize=(12,5),kind='bar',color='green');

plt.xlabel('Sentiment')

plt.ylabel('Total Number Of Individual Sentiment for Training')
lens = [len(x) for x in train.text]

plt.figure(figsize=(12, 5));

print (max(lens), min(lens), np.mean(lens))

sns.distplot(lens);

plt.title('Text length distribution')
lens = [len(x) for x in train.selected_text]

plt.figure(figsize=(12, 5));

print (max(lens), min(lens), np.mean(lens))

sns.distplot(lens);

plt.title('Text length distribution')
from sklearn.feature_extraction import text

stop_words = text.ENGLISH_STOP_WORDS

train['text']=train['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

train.head(10)
X_train=train['text']+' '+ train['sentiment']

y_train=train['selected_text']
X_test= test['text']+' '+ test['sentiment']
import nltk

import re

import string



from nltk.sentiment.vader import SentimentIntensityAnalyzer

test_subset=test['text']

sentiment= test['sentiment']



sid = SentimentIntensityAnalyzer()

word_list=[]

i=0

for word in test_subset:

        #Removing URL

        word = re.sub('http[s]?://\S+', '', word)

        split_text= word.split()

        #Removing Punctuation

        #word=re.sub('[!#?,.:"-;]', '', word)

        

        score_list=[]

        

        if sentiment[i]=='positive':

            for w in split_text:

                score=sid.polarity_scores(w)['compound']

                score_list.append(score)

                max_index=np.argmax(score_list)

            word_list.append(split_text[max_index])

                    

        elif sentiment[i]=='negative':

            for w in split_text:

                score=sid.polarity_scores(w)['compound']

                score_list.append(score)

                min_index=np.argmin(score_list)

            word_list.append(split_text[min_index])

         

        else:

             word_list.append(word)

                

        i=i+1

       

 
submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

submission["selected_text"]= word_list
submission.head(10)
submission.to_csv('submission.csv', index=False)