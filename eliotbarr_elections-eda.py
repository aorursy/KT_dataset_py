# imports

import numpy as np

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

import sqlite3

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

french_stemmer=SnowballStemmer('french')

import re

from wordcloud import WordCloud



%matplotlib inline



import matplotlib

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (16,10)

import seaborn as sns
# Read sqlite query results into a pandas DataFrame

connection = sqlite3.connect("../input/database.sqlite")

dataframe = pd.read_sql_query("SELECT * from data", connection)



# close sqlite connection

connection.close()
dataframe.head()
dataframe['timestampms'] = pd.to_numeric(dataframe.timestampms)

dataframe['datetime'] = pd.to_datetime(dataframe['timestampms'], unit='ms')



dataframe['year'] = dataframe.datetime.dt.year

dataframe['month'] = dataframe.datetime.dt.month

dataframe['day'] = dataframe.datetime.dt.day

dataframe['hour'] = dataframe.datetime.dt.hour

dataframe['minute'] = dataframe.datetime.dt.minute
dataframe["text"].loc[1]
def review_to_wordlist( review, remove_stopwords=True, stemmer=True):

    # Function to convert a document to a sequence of words,

    # optionally removing stop words.  Returns a list of words.



    review_text = review.replace("@",'')

    words = review_text.lower().split()

    

    wordcloud = review_text.lower().split()



    if remove_stopwords:

        stops = set(stopwords.words("french"))

        stops = list(stops) + ['https', 'co']

        words = [w for w in words if not w in stops]

        

    if stemmer:

        b = []

        stemmer = french_stemmer

        for word in words:

            b.append(stemmer.stem(word))

    else:

        b = words

    # 5. Return a list of words

    return(b)
review_to_wordlist(dataframe["text"].loc[1], stemmer=False)
clean_train_reviews = []

for review in dataframe['text'].loc[:10000]:

    clean_train_reviews.append( " ".join(review_to_wordlist(review, stemmer = False)))
cs = ""

for i in clean_train_reviews:

    cs = cs + " " + i

stops_for_wc = stopwords.words("french") + ['https','co','rt']



wc = WordCloud(max_words=2000,

               random_state=1, stopwords = stops_for_wc).generate(cs)



plt.imshow(wc)

plt.axis("off")

plt.show()