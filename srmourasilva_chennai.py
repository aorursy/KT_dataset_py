from collections import Counter

import pandas as pd

import nltk as nltk

import numpy as np

from nltk import word_tokenize

from sklearn.model_selection import train_test_split

import string
chennai = pd.read_csv('../input/chennai_reviews.csv')

chennai.head()
descriptions = chennai['Review_Text']



translator = str.maketrans('', '', string.punctuation)



def decode_string(s):

    s = s.translate(translator)

    return word_tokenize(s)



#word_count_dict = [dict(Counter(decode_string(description))) for description in descriptions]

word_count_dict = [

    {word : 1 for word in decode_string(description)} for description in descriptions

]
categories = chennai['Sentiment']



featureList = []



for word_dict, category in zip(word_count_dict, categories):

    featureList.append((word_dict, category))

    

featureList[0]