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

import eli5

import matplotlib.pylab as plt

#загрузка дополнительных данных для nltk

import nltk

#nltk.download('punkt')

#nltk.download('stopwords')

#nltk.download('wordnet')

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score, classification_report, f1_score

from sklearn.model_selection import RandomizedSearchCV

from nltk.stem import SnowballStemmer, WordNetLemmatizer, LancasterStemmer

from functools import lru_cache



df = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')

test = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')
vectorizer = TfidfVectorizer()

sklearn_tokenizer = vectorizer.build_tokenizer()



LEMMATIZER = WordNetLemmatizer()



@lru_cache(maxsize=2048)

def lemmatize_word(word):

    parts = ['a','v','n','r']

    for part in parts:

        temp = LEMMATIZER.lemmatize(word, part)

        if temp != word:

            return temp

    return word    



def lemm_question(question):

    return list(lemmatize_word(w.lower()) for w in sklearn_tokenizer(question))



stemmer = SnowballStemmer('english')



def stem_question(question):

    return list(stemmer.stem(w) for w in sklearn_tokenizer(question))



df['stem'] = df.apply (lambda row: " ".join(stem_question(row.question_text)),axis=1)

test['stem'] = test.apply (lambda row: " ".join(stem_question(row.question_text)),axis=1)

df['lemm'] = df.apply (lambda row: " ".join(lemm_question(row.question_text)),axis=1)

test['lemm'] = test.apply (lambda row: " ".join(lemm_question(row.question_text)),axis=1)

#EMBEDDING_FILE = '/kaggle/input/quora-insincere-questions-classification/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

#w2v = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

#w2v = KeyedVectors.load_word2vec_format(imgdata, binary=True)
stops=stopwords.words('english')

vec = TfidfVectorizer(stop_words = stops)

clf = SGDClassifier(loss='modified_huber',class_weight={0:1,1:11})

model1 = Pipeline([('vec', vec),('clf', clf)])

model1.fit(df.stem.values, df.target.values)

preds1 = model1.predict_proba(test.stem.values)
stops=stopwords.words('english')

vec = TfidfVectorizer(stop_words = stops)

clf = SGDClassifier(loss='modified_huber',class_weight={0:1,1:9})

model2 = Pipeline([('vec', vec),('clf', clf)])

model2.fit(df.question_text.values, df.target.values)

preds2 = model2.predict_proba(test.question_text.values)
#stops=stopwords.words('english')

vec = TfidfVectorizer()

clf = SGDClassifier(loss='log',class_weight={0:1,1:11})

model3 = Pipeline([('vec', vec),('clf', clf)])

model3.fit(df.lemm.values, df.target.values)

preds3 = model3.predict_proba(test.lemm.values)

preds = (preds1[:,1]+preds2[:,1]+preds3[:,1])/3



# можно поискать идеальную границу, но она тут всегда около 0.7, так что..

mysub=pd.DataFrame({'qid':test.qid,'prediction':preds>=0.7})

mysub['prediction']=mysub['prediction'].astype(np.int64)

mysub.to_csv("submission.csv",index=False)