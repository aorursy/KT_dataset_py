# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt

import re

from sklearn.metrics import accuracy_score

import nltk

from nltk.corpus import stopwords

from nltk import regexp_tokenize

from nltk.stem import WordNetLemmatizer

import spacy





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv(r'/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv(r'/kaggle/input/nlp-getting-started/test.csv')

train.head()
nlp = spacy.load('en_core_web_lg')
with nlp.disable_pipes():

    train_vectors = np.array([nlp(text).vector for text in train.text])

    test_vectors = np.array([nlp(text).vector for text in test.text])



print(train_vectors.shape, test_vectors.shape)
X_train = train_vectors

y_train = train.target.to_numpy()



train_x, test_x, train_y, test_y = train_test_split(X_train, y_train, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

svc = SVC(kernel='rbf', C=0.7, gamma='auto', probability=True)

rfc = RandomForestClassifier(n_estimators=100)

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
# class keras_model:

#     def __call__():

#         model = keras.models.Sequential()

#         model.add(keras.layers.Dense(1024, activation='relu'))

#         model.add(keras.layers.Dense(512, activation='relu'))

#         model.add(keras.layers.BatchNormalization())

#         model.add(keras.layers.Dense(512, activation='relu'))

#         model.add(keras.layers.Dropout(0.2))

#         model.add(keras.layers.Dense(128, activation='relu'))

#         model.add(keras.layers.Dense(128, activation='relu'))

#         model.add(keras.layers.Dense(1, activation='sigmoid'))

#         model.compile(optimizer='adam',

#                      loss='binary_crossentropy',

#                      metrics=['accuracy'])

#         return model



# from keras.wrappers.scikit_learn import KerasClassifier

# keras_clf = KerasClassifier(keras_model(), batch_size=100, epochs=20)

# model.fit(X_train, y_train, batch_size=100, epochs=40)



# preds = model.predict_classes(test_vectors)
vcf = VotingClassifier(estimators=[('svc', svc), ('rfc', rfc), ('gbc', gbc),], voting='soft')

vcf.fit(X_train, y_train)
preds = vcf.predict(test_vectors)

print(accuracy_score((vcf.predict(test_x)), test_y))

print(len(test['id']), len(preds))
submission = pd.DataFrame(columns=['id', 'target'])

submission['id'] = test['id']

submission['target'] = preds

submission
submission.to_csv('submission.csv', index=False)