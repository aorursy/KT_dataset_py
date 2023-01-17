%matplotlib inline



import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import numpy as np



from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cross_validation import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.porter import PorterStemmer

import re

import nltk

from nltk.corpus import stopwords



con = sqlite3.connect('../input/database.sqlite')



messages = pd.read_sql_query("""

SELECT Score, Text

FROM Reviews

WHERE Score != 3

""", con)



def review_to_words( raw_review ):

    # Function to convert a raw review to a string of words

    # The input is a single string (a raw movie review), and 

    # the output is a single string (a preprocessed food review)

    #

    # 1. Remove non-letters        

    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 

    #

    # 2. Convert to lower case, split into individual words

    words = letters_only.lower().split()                             

    #

    # 3. In Python, searching a set is much faster than searching

    #   a list, so convert the stop words to a set

    stops = set(stopwords.words("english"))                  

    # 

    # 4. Remove stop words

    meaningful_words = [w for w in words if not w in stops]   

    #

    # 5. stemming the remaining words

    stemmer = nltk.PorterStemmer()

    stemmedWords = []

    for word in meaningful_words:

        stemmedWords.append(stemmer.stem(word))

        

    # 5. Join the words back into one string separated by space, 

    # and return the result.

    return( " ".join(stemmedWords))
print(review_to_words(messages["Text"][1]))
text = messages["Text"].apply(review_to_words)
messages["Text"] = text
messages.to_csv("reviews.csv",index=False)