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
df = pd.read_csv('../input/spam.csv', encoding = 'ISO-8859-1')
df.head(5)
df.head(5)['v2'].values
import nltk

def get_word_set(sentences):
    word_set = set()
    for sens in sentences:
        words = nltk.word_tokenize(sens.lower())
        word_set.update(words)
    return word_set

word_set = get_word_set(df['v2'].values)
def words_to_feature_vector(word_set, sens):
    feature_vector = list()
    words = get_word_set(sens)
    for word in word_set:
        if word in words:
            feature_vector.append(1)
        else:
            feature_vector.append(0)
    return feature_vector

X = list()
Y = list()

for index, row in df.iterrows():
    sens = row['v2'].lower()
    label = row['v1']
    feature_vector = words_to_feature_vector(word_set, sens)
    
    X.append(feature_vector)
    Y.append(label)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)
clf.score(X, Y)
training_prediction = clf.predict(X)
df['predict1'] = training_prediction
df[df['v1'] != df['predict1']]
from sklearn.naive_bayes import MultinomialNB
clf1 = MultinomialNB()
clf1.fit(X, Y)
clf1.score(X, Y)
training_prediction = clf1.predict(X)
df['predict2'] = training_prediction
df[df['v1'] != df['predict2']]
