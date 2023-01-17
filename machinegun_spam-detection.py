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
df = pd.read_csv("../input/spam.csv",  encoding = "ISO-8859-1")
df.head()
from sklearn.utils import shuffle
df = shuffle(df)
df.shape
import matplotlib.pyplot as plt
df.v1.value_counts().plot(kind="bar")
plt.show()
import re
import string


#removes everything apart from text 
def clean_text(s):
    s = re.sub("[^a-zA-Z]", " ",s)
    s = re.sub(' +',' ', s)        
    return s

df['Text'] = [clean_text(s) for s in df['v2']]
df.head()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from nltk.stem import SnowballStemmer

#stemming support
english_stemmer = SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc:([english_stemmer.stem(w) for w in analyzer(doc)])

# pull the data into vectors
#tokenize words, remove stop words and stem
vectorizer = StemmedCountVectorizer(analyzer='word',stop_words='english')
#vectorizer = CountVectorizer(stop_words='english')
x = vectorizer.fit_transform(df['Text'])
#vectorizer.vocabulary_
x.shape
encoder = LabelEncoder()
y = encoder.fit_transform(df['v1'])

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# take a look at the shape of each of these
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from numpy import argmax
from sklearn.preprocessing import normalize

#batch_size = 10
epochs = 4

nn_model = Sequential()
nn_model.add(Dense(2000, input_dim=len(vectorizer.vocabulary_), activation='relu'))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(500, activation='relu'))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(1, activation='sigmoid'))

nn_model.summary()

nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])


# Fit the model
nn_model.fit(x_train.toarray(), y_train, verbose=1,
                        epochs=epochs)

scores = nn_model.evaluate(x_test.toarray(), y_test)
print(scores[1])