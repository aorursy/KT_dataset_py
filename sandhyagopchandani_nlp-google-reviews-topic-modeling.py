# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

nltk.download('wordnet')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
app_store = pd.read_csv('../input/googleplaystore.csv')

playstore_user_reviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')
#playstore_user_reviews



sum_missing = playstore_user_reviews.isnull().sum()

percent_missing = playstore_user_reviews.isnull().sum() * 100 / len(playstore_user_reviews)



missing_value_df = pd.DataFrame({'column_name': playstore_user_reviews.columns,

                                 'total_missing':sum_missing,

                                 'percent_missing': percent_missing})

missing_value_df
playstore_user_reviews = playstore_user_reviews[playstore_user_reviews['Translated_Review'].notnull()].reset_index(drop=True)
sum_missing = playstore_user_reviews.isnull().sum()

percent_missing = playstore_user_reviews.isnull().sum() * 100 / len(playstore_user_reviews)



missing_value_df = pd.DataFrame({'column_name': playstore_user_reviews.columns,

                                 'total_missing':sum_missing,

                                 'percent_missing': percent_missing})

missing_value_df
playstore_user_reviews.groupby('Sentiment').size()

playstore_user_reviews.Sentiment=[0 if i=="Positive" else 1 if i== "Negative" else 2 for i in playstore_user_reviews.Sentiment]

#df.head(10)
compact_app_reviews = playstore_user_reviews.groupby(['App'])['Translated_Review'].apply(lambda x: ','.join(x)).reset_index()

#complete_data = pd.merge(app_store, compact_app_reviews, on='App', how='left')

print(compact_app_reviews.head(10))

compact_app_reviews.shape
import spacy

spacy.load('en')

from spacy.lang.en import English

parser = English()

def tokenize(text):

    lda_tokens = []

    tokens = parser(text)

    for token in tokens:

        if token.orth_.isspace():

            continue

        elif token.like_url:

            lda_tokens.append('URL')

        elif token.orth_.startswith('@'):

            lda_tokens.append('SCREEN_NAME')

        else:

            lda_tokens.append(token.lower_)

    return lda_tokens


from nltk.corpus import wordnet as wn

def get_lemma(word):

    lemma = wn.morphy(word)

    if lemma is None:

        return word

    else:

        return lemma

    

from nltk.stem.wordnet import WordNetLemmatizer

def get_lemma2(word):

    return WordNetLemmatizer().lemmatize(word)
nltk.download('stopwords')

en_stop = set(nltk.corpus.stopwords.words('english'))
def prepare_text_for_lda(text):

    tokens = tokenize(text)

    tokens = [token for token in tokens if len(token) > 4]

    tokens = [token for token in tokens if token not in en_stop]

    tokens = [get_lemma(token) for token in tokens]

    return tokens