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
test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

submission=pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
import nltk

import re

import string

import heapq

nltk.download("stopwords")

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords 



test_subset=test['text']

sentiment= test['sentiment']



sid = SentimentIntensityAnalyzer()

stop_words = set(stopwords.words('english')) 



word_list=[]

sen_list=[]

k=0

no_of_elem=1

for word in test_subset:

    #Removing URL

    word = re.sub('http[s]?://\S+', '', word)

    split_text= word.split()

    split_text = [w for w in split_text if not w in stop_words]

    score_list=[]

  

    if sentiment[k]=='positive':

        for w in split_text:

            score=sid.polarity_scores(w)['compound']

            score_list.append(score)

        for i in [x for x in set(heapq.nlargest(no_of_elem,score_list))]:

            word_list.append(split_text[score_list.index(i)])

        sen_list.append((" ".join(word_list)).strip())

        word_list.clear()

                

    elif sentiment[k]=='negative':

        for w in split_text:

            score=sid.polarity_scores(w)['compound']

            score_list.append(score)

        for i in [x for x in set(heapq.nsmallest(no_of_elem,score_list))]:

            word_list.append(split_text[score_list.index(i)])

        sen_list.append((" ".join(word_list)).strip())

        word_list.clear()

      

    else:

        sen_list.append((word).strip())

    k=k+1
submission['selected_text'] = sen_list
submission.head(5)
submission.to_csv('submission.csv', index=False)