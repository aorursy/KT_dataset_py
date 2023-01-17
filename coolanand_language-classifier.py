# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import pandas as pd

import numpy as np

import string

import re

import json

import matplotlib.pyplot as plt

%matplotlib inline



import nltk

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score



from nltk.corpus import stopwords

nltk_stopwords = stopwords.words('english')



remove_punctuation = '!"$%&\'()*+,-./:;<=>?@[\\]“”^_`{|}~’'

from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.utils import np_utils



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

english_text = pd.read_csv("../input/english_text.csv")

hinglish_text = pd.read_csv("../input/hinglish_text.csv")
english_text.head()
hinglish_text.head()
print(english_text.shape)

print(hinglish_text.shape)
hinglish_text.isna().sum()
def clean_column(dataframe, column_to_clean, new_col):

    df_copy = dataframe.copy()

    df_copy['copied_column'] = df_copy[column_to_clean]

    df_copy['copied_column'] = df_copy['copied_column'].str.lower()

    cleaned_column = []

    for label in df_copy.index:

        row = df_copy.loc[label, :]['copied_column']

        clean = [x for x in row.split() if x not in string.punctuation]

        clean = [x for x in clean if x not in nltk_stopwords]

        clean = [x for x in clean if x not in string.digits]

        clean = [x for x in clean if x not in remove_punctuation]

        clean = [x for x in clean if len(x) != 1]

        clean = " ".join(clean)

        clean = clean.strip()

        cleaned_column.append(clean)

    df_copy[new_col] = cleaned_column

    del df_copy['copied_column']

    return df_copy
english_text_copy = clean_column(english_text, 'text', 'clean_text')

english_text_copy.drop(['text'], axis=1, inplace = True)
english_text_copy
hinglish_text_copy = clean_column(hinglish_text, 'text', 'clean_text')

hinglish_text_copy.drop(['text'], axis=1, inplace = True)
hinglish_text_copy
english_text_labels = np.zeros((english_text.shape[0],1), dtype=np.int16)

hinglish_text_labels = np.ones((hinglish_text.shape[0],1), dtype=np.int16)
english_text_copy['labels'] = english_text_labels

hinglish_text_copy['labels'] = hinglish_text_labels
english_text_copy = english_text_copy.take(np.random.permutation(len(english_text_copy))[:4470])
english_text_copy.shape
copy_df = english_text_copy.append(hinglish_text_copy)
copy_df
copy_df.drop(['ID'], axis=1, inplace=True)
copy_df
X_train, X_test, y_train, y_test = train_test_split(copy_df['clean_text'], copy_df['labels'], test_size=0.33, random_state=6001)
y_train
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features= 10000,strip_accents='unicode', norm='l2')
x_train = vectorizer.fit_transform(X_train).todense()

x_test = vectorizer.transform(X_test).todense()
np.random.seed(1337)

batch_size = 64

nb_epochs = 20
y_test
Y_train = np_utils.to_categorical(y_train, 2)

Y_test = np_utils.to_categorical(y_test, 2)
Y_test
import gensim 

from gensim.models import Word2Vec 
model = Sequential()

model.add(Dense(1000,input_shape= (10000,)))

model.add(Activation('tanh'))

model.add(Dropout(0.5))

model.add(Dense(500))

model.add(Activation('tanh'))

model.add(Dropout(0.5))

model.add(Dense(50))

model.add(Activation('tanh'))

model.add(Dropout(0.5))

model.add(Dense(2))

model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

print (model.summary())
y_train.shape
model.fit(x_train, Y_train, batch_size=batch_size, epochs=20,verbose=1)
y_train_pred = model.predict_classes(x_train,batch_size=batch_size)

y_test_pred = model.predict_classes(x_test,batch_size=batch_size)
y_train_pred
# Validation accuracy

accuracy_score(y_train, y_train_pred)*100
# Testing accuracy

accuracy_score(y_test, y_test_pred)*100