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

import scipy.io

from array import *

import numpy as np

import re

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import re, string

import nltk

from nltk import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from nltk.stem import LancasterStemmer, WordNetLemmatizer

nltk.download('stopwords')

nltk.download('wordnet')

nltk.download('vader_lexicon')

import string

from string import digits

STOPWORDS = set(stopwords.words('english'))
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv", dtype=str)

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv', dtype=str)

sub = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
train.head()
test.head()
sub.head()
def clean_text(text):

    ## Remove puncuation

    #text = text.translate(string.punctuation)

    text = str(text)

    text= text.lower()

    ## Convert words to lower case and split them

    url = re.compile(r'https?://\S+|www\.\S+')

    text= url.sub(r'',text)

    html=re.compile(r'<.*?>')

    text= html.sub(r'',text)

    remove_digits = str.maketrans('', '', digits)

    text = text.translate(remove_digits)

    ## Remove stop words

    #text=" ".join([word for word in str(text).split() if word not in STOPWORDS])

    return text
train["text"]=train["text"].apply(clean_text)

test["text"]=test["text"].apply(clean_text)
train.head()
def choosing_selectedword(df_process):

    train_text = df_process['text']

    train_sentiment = df_process['sentiment']

    selected_text_processed = []

    analyser = SentimentIntensityAnalyzer()

    for j in range(0 , len(train_text)):

        text = str(train_text.iloc[j])

        # For Neutral append the full sentence

        if(train_sentiment.iloc[j] == "neutral"):

            selected_text_processed.append(str(text))

        #For positive take only words with positive polarity

        if(train_sentiment.iloc[j] == "positive"):

            token = re.split(' ', text)

            ss_arr = ""

            polar = 0

            for word in token:

                score = analyser.polarity_scores(word)

                if score['compound'] >polar:

                    polar = score['compound']

                    ss_arr = ss_arr + " "+word

            if len(ss_arr) != 0:

                selected_text_processed.append(ss_arr)   

            if len(ss_arr) == 0:

                selected_text_processed.append(text)

        #for negative take words with negative polarity 

        if(train_sentiment.iloc[j] == "negative"):

            token = re.split(' ', text)

            ss_arr = ""

            polar = 0

            for word in token:

                score = analyser.polarity_scores(word)

                if score['compound'] <polar:

                    polar = score['compound']

                    ss_arr = ss_arr + " " + word

            if len(ss_arr) != 0:

                selected_text_processed.append(ss_arr)   

            if len(ss_arr) == 0:

                selected_text_processed.append(text)  

    return selected_text_processed
train["predicted"]=choosing_selectedword(train)
train.head()
test["selected_text"]= choosing_selectedword(test)

sub["selected_text"]= choosing_selectedword(test)
sub.head()
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
average = 0

for i in range(0,train.shape[0]):

    jaccard_score = jaccard(str(train["selected_text"][i]),str(train["predicted"][i]))

    average += jaccard_score 

print('Training Data average jaccard score is ', average/len(train["selected_text"]))
sub.to_csv('submission.csv', index = False)