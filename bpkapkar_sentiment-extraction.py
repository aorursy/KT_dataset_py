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

import re

import os



train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sub = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')



sub['selected_text'] = test['text']

sub['sentiment'] = test['sentiment']
import nltk

from nltk.tokenize import word_tokenize

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()



for b, i in enumerate(sub['selected_text']):

    if sub['sentiment'].iloc[[b]].values == ['positive']:

        pos_word_list=[]

        word_tokens = word_tokenize(i)

        for word in word_tokens:

            if (sid.polarity_scores(word)['compound']) >= 0.01:

                pos_word_list.append(word)

        pos_word_list = " ".join(pos_word_list)

        sub['selected_text'].iloc[[b]] = pos_word_list
sid = SentimentIntensityAnalyzer()



for b, i in enumerate(sub['selected_text']):

    if sub['sentiment'].iloc[[b]].values == ['negative']:

        neg_word_list=[]

        word_tokens = word_tokenize(i)

        for word in word_tokens:

            if (sid.polarity_scores(word)['compound']) <= -0.01:

                    neg_word_list.append(word)

        neg_word_list = " ".join(neg_word_list)

        sub['selected_text'].iloc[[b]] = neg_word_list

del sub['sentiment']

sub.to_csv("submission.csv", index=None)

sub.head()