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

from nltk.tokenize import word_tokenize

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords
test.head(20)
vader = SentimentIntensityAnalyzer()
vader.polarity_scores('happy')
word_tokenize(re.sub('\W+',' ',test['text'][2379]))
vader.polarity_scores('m')
total_words = []

for i in range(len(test)):

    text = test['text'][i]

    text = re.sub('http[s]?://\S+', '', text)

    text = re.sub('\W+',' ',text)

    words = word_tokenize(text)

    #words = [w for w in words if w not in stop_words]

    

    score = []

    

    if test['sentiment'][i] == 'positive':

        

        for w in words:

            score.append(vader.polarity_scores(w)['compound'])

            

        maximum = np.argmax(score)

        word = words[maximum]

        total_words.append(word)

        

    if test['sentiment'][i] == 'negative':

        

        for w in words:

            score.append(vader.polarity_scores(w)['compound'])

       

        maximum = np.argmin(score)

        word = words[maximum]

        total_words.append(word)

        

    if test['sentiment'][i] == 'neutral':

        

        total_words.append(text)
submission.head()
submission['selected_text'] = total_words
submission.head(20)
submission.to_csv('submission.csv',index = False)