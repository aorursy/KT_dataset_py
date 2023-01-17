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
# Importing the required libraries

import numpy as np

import pandas as pd

import nltk

import re

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Reading the train,test and submission file into different dataframes

train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sample_sub = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
sia = SentimentIntensityAnalyzer()
train.head(20)
# Creating an empty list to store the words

total_words = []



for i in range(len(test)):

    # Reading the text columnn from the test dataframe

    text = test['text'][i]

    # Creating an empty list to store the scores

    score = []

    

    # Checking for text with a positive sentiment allocated to it

    if test['sentiment'][i] == 'positive':

        # Removing https tags

        text = re.sub(r'http\S+', '', text)

        # Getting individual words from the text. Using re.split() helps retain the punctuations like '!' mark

        words = re.split(' ', text)

        # Looping over the words to get polarity scores for each word

        for w in words:

            score.append(sia.polarity_scores(w)['compound'])

        # Finding the word with the maximum polarity score and storing it

        maximum = np.argmax(score)

        word = words[maximum]

        total_words.append(word)



    # Checking for text with a negative sentiment allocated to it    

    if test['sentiment'][i] == 'negative':

        # Removing https tags

        text = re.sub(r'http\S+', '', text)

        # Getting individual words from the text. Using re.split() helps retain the punctuations like '!' mark

        words = re.split(' ', text)

        # Looping over the words to get polarity scores for each word

        for w in words:

            score.append(sia.polarity_scores(w)['compound'])

        # Finding the word with the minimum polarity score and storing it

        minimum = np.argmin(score)

        word = words[minimum]

        total_words.append(word)

    # Checking for text with a neutral sentiment allocated to it     

    if test['sentiment'][i] == 'neutral':

        # For a neutral sentiment text, it is observed that the entire text is considered for selected_text. Hence this code also performs the same operation 

        total_words.append(text)
# Having a look at the total words

total_words
# Setting the selected text in the sample submission dataframe as per the total_words list

sample_sub['selected_text'] = total_words
# Having a look at the sample_sub dataframe

sample_sub.head(20)
# Saving the output into the submission.csv file as per the competition requirements

sample_sub.to_csv('submission.csv',index = False)