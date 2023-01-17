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

import numpy as np

import re

from tqdm import tqdm



from nltk.corpus import stopwords



from gensim.models import Word2Vec

from gensim.models import Phrases

from gensim.models.phrases import Phraser



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



from keras.utils import np_utils

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, Embedding, Dropout,Bidirectional, Reshape, Flatten, CuDNNGRU, CuDNNLSTM

from keras.models import Model, Sequential

from keras.initializers import Constant

from keras import backend as K
def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        """Precision metric.



        Only computes a batch-wise average of precision.



        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]



def clean_text(x):

    x = str(x)

    for punct in puncts:

        x = x.replace(punct,"")

        

        

clean_text([train_df,test_df])

train_df['lower_questions'] = train_df['question_text'].apply(lambda x: x.lower())

test_df['lower_questions'] = test_df['question_text'].apply(lambda x: x.lower())

train_X = train_df['lower_questions']

test_X = test_df['lower_questions']
import warnings

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer

# TFIDF 설정

tfv = TfidfVectorizer()



# Fit TFIDF (훈련)

tfv.fit(pd.concat([train_X, test_X])) # Learn vocabulary and idf from training set.



# 변환

X =  tfv.transform(train_X)

X_test = tfv.transform(test_X)



train_y = train_df['target'].values

#train_y = np_utils.to_categorical(train_y)

print(X.shape)

print(X_test.shape)

print(train_y.shape)
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(X, train_y)

y_test_predict = clf.predict(X_test)
test_df = pd.read_csv("../input/test.csv", usecols=["qid"])

out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = y_test_predict

out_df.to_csv("submission.csv", index=False)