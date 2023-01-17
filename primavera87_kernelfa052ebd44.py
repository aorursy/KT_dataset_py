# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd



initial_data = pd.read_csv('../input/twitter_kabir_singh_bollywood_movie.csv', delimiter=',')

initial_data.head()
initial_data['author'].value_counts()
text = initial_data['text_raw']

import nltk.classify.util

from nltk.classify import NaiveBayesClassifier

from nltk.corpus import names



def word_feats(words):

    return dict([(word, True) for word in words])

 

positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great']

negative_vocab = [ 'bad', 'terrible','useless', 'hate']

neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not']



positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]

negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]

neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]



train_set = positive_features + negative_features + neutral_features



classifier = NaiveBayesClassifier.train(train_set)
total_text = pd.DataFrame()

total_text['text'] = text

total_text['sentiment'] = 'neu'





for i in range(total_text.shape[0]):

    words = total_text['text'][i].split(' ')

    pos = 0

    neg = 0

    for word in words:

        classResult = classifier.classify(word_feats(word))

        if classResult == 'neg':

            neg = neg + 1

        if classResult == 'pos':

            pos = pos + 1

    if pos > neg:

        total_text['sentiment'][i] = 'pos'

    if neg > pos:

        total_text['sentiment'][i] = 'neg'

        

print(total_text.head())
total_text['sentiment'].value_counts()
total_text['favorite_count'] = initial_data['favorite_count']

total_text['reply_count'] = initial_data['reply_count']



total_text.sort_values(by=['favorite_count'], ascending = False).head(10)
total_text.loc[total_text['sentiment']=='neg'].sort_values(by=['favorite_count'], ascending = False).head(10)
total_text.sort_values(by=['reply_count'], ascending = False).head(10)
total_text.loc[total_text['sentiment']=='neg'].sort_values(by=['reply_count'], ascending = False).head(10)