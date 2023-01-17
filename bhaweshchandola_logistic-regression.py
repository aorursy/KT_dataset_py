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

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier

from scipy.sparse import hstack, vstack

import csv

import re

from nltk.corpus import stopwords

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier

import gensim

from gensim.models import KeyedVectors

from gensim.test.utils import datapath

from sklearn.ensemble import RandomForestRegressor

import nltk

import string

# !pip install contractions

# import contractions

# nltk.download('stopwords')

# nltk.download('wordnet')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^a-z ]')

STOPWORDS = set(stopwords.words('english'))

stemmer = nltk.stem.WordNetLemmatizer()



def text_prepare(text):

    """

        text: a string

        

        return: modified initial string

    """

    text = text.lower()# lowercase text

#     text = " ".join([contractions.fix(t) for t in text.split(' ')])

    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)# replace REPLACE_BY_SPACE_RE symbols by space in text

    text = re.sub(BAD_SYMBOLS_RE, '', text)# delete symbols which are in BAD_SYMBOLS_RE from text

    text = ' '.join([word.strip(string.punctuation) for word in text.split(' ')])

    text = ' '.join([t for t in text.split(" ") if len(t) > 2])

    text = ' '.join([stemmer.lemmatize(x) for x in text.split() if x not in STOPWORDS])# delete stopwords from text

    return text

    

    
df = pd.read_csv("/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv", sep="\t")

df.head()
print(df.iloc[0]["Phrase"])

print(df["Sentiment"].value_counts())
# sample run of function

text_prepare(df.iloc[0]["Phrase"])
df["Phrase"] = df["Phrase"].apply(text_prepare)
# tfidf = TfidfVectorizer(ngram_range=(1,2))

# features = tfidf.fit_transform(np.array(train_x)) #fit transform to learn vocubulary and transform in matrix



trainx = df["Phrase"]

trainy = df["Sentiment"]



tfidf = TfidfVectorizer(min_df=5, max_df=0.5, ngram_range=(1,3))

features = tfidf.fit_transform(np.array(trainx)) #fit transform to learn vocubulary and transform in matrix





regressor = OneVsRestClassifier(LogisticRegression())



regressor.fit(features, np.array(trainy))

# regressor.fit(np.array(question2vec_result), np.array(train_y))
df1 = pd.read_csv("/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv", sep="\t")

df1.head()
df1["Phrase"] = df1["Phrase"].apply(text_prepare)

testx = df1["Phrase"]



features3 = tfidf.transform(np.array(testx))





x3 = regressor.predict(features3)



x4 = pd.DataFrame({'PhraseId':np.array(df1["PhraseId"]), 'Sentiment':x3})

x4["Sentiment"] = x4["Sentiment"].apply(lambda x:int(round(x)))

x4.to_csv("output.csv", index=False)