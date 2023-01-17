import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import re

import time

from sklearn.model_selection import train_test_split

import spacy

from sklearn.metrics import accuracy_score

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

embeddings = nlp.vocab.vectors.data

from nltk.corpus import stopwords

from nltk import word_tokenize

STOPWORDS = set(stopwords.words('english'))
df = pd.read_csv('/kaggle/input/amazonreviews/amazon2.csv')
df.head()
df['reviewText'] = df['summary'].map(str) + " " + df['reviewText'].map(str)

df = df.drop('summary',axis=1)

gc.collect()
df.head()
df = df.reset_index(drop=True)



REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))



def clean_text(text):

    """

        text: a string

        

        return: modified initial string

    """

    text = text.lower() # lowercase text

    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.

    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 

    text = text.replace('x', '')

#    text = re.sub(r'\W+', '', text)

    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text

    return text

df['reviewText'] = df['reviewText'].astype('str').apply(clean_text)

df['reviewText'] = df['reviewText'].str.replace('\d+', '')
df.head()
#Below. I am converting the "reviewText" column string into numbers using Glove model. 

label = df['overall']

df =  np.matrix(df.reviewText.to_dense().apply(lambda x: nlp(x).vector).tolist())

gc.collect()
gc.collect()
train, test, label, label_test = train_test_split(df,label, random_state=42, test_size=0.2)
del df

gc.collect()
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(train, label)
pred = model.predict(test)
print("Accuracy:", accuracy_score(label_test, pred))