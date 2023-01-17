# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from tensorflow import keras

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import time

from sklearn import tree



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



df = pd.read_csv('../input/ireland-historical-news/irishtimes-date-text.csv')



category = df.headline_category.unique()

df=df.replace(to_replace=category, value=[c.split('.')[0] for c in category])



category = df.headline_category.unique()

print(category)



cat_index = dict(np.transpose([category, np.arange(len(category))]))

index_cat = {v: k for k, v in cat_index.items()}



all_word = {cat:{} for cat in category}

word_index = {"SP":0, "UNK":1}

max_length = 0

for [headline, cat] in df[['headline_text', 'headline_category']].to_numpy():

    l = headline.split()

    n = len(l)

    if n>max_length:

        max_length=n

    for word in l:

        if word not in all_word[cat]:

            all_word[cat][word] =  0

        else:

            all_word[cat][word] += 1



for cat in all_word:

    all_word[cat] = sorted(all_word[cat], key=lambda word:all_word[cat][word], reverse=True)[:1000]

    

def not_in_all(word, l):

    for li in l:

        if word not in li: return True

    return False



i=2

for cat in all_word.values():

    for word in cat:

        if word not in word_index:

            if not_in_all(word, all_word.values()):

                word_index[word] = i

                i += 1

print('ditionnary size', len(word_index))
def get_word_index(word):

    return word_index[word]



def get_sentence_index(sentence):

    out = np.zeros(max_length)

    for (i, word) in enumerate(sentence.split()):

        if word in word_index:

            out[i] = get_word_index(word)

        else :

            out[i] = 1

    return out



def get_sentence_occurrence(sentence):

    out = np.zeros(len(word_index))

    for (i, word) in enumerate(sentence.split()):

        if word in word_index:

            out[get_word_index(word)] += 1

        else :

            out[1] += 1

    return out

            

def get_indexofcat(category):

    i = cat_index[category]

    return i



def get_catofindex(i):

    category = index_cat[i]

    return category



print(max_length)

print("préprocessing data ......................")

df = df.sample(frac=1).reset_index(drop=True)

train_set = np.array([get_sentence_occurrence(headline) for headline in df.headline_text[:200000]])

train_labels = np.array([get_indexofcat(category) for category in df.headline_category[:200000]])



val_set = np.array([get_sentence_occurrence(headline) for headline in df.headline_text[200000:250000]])

val_labels = np.array([get_indexofcat(category) for category in df.headline_category[200000:250000]])

print("....................... Data preprocessed")



def count(t):

    out = np.zeros(len(cat_index))

    for i in t:

        out[i] += 1

    return out



plt.figure(figsize=(8,6))

plt.bar(index_cat.values(), count(train_labels), color=['g', 'b', 'y', 'r', 'c', 'm'])

plt.xticks(rotation=90)

plt.show()
model = tree.DecisionTreeClassifier(min_samples_leaf=2)

model = model.fit(train_set, train_labels)



print("score on training set : ",model.score(train_set, train_labels))

print("score on validation set : ",model.score(val_set, val_labels))
from sklearn.ensemble import RandomForestClassifier



rfmodel = RandomForestClassifier(min_samples_leaf=2, n_estimators=100, verbose=1, n_jobs=-1)

print(len(train_set))

rfmodel = rfmodel.fit(train_set, train_labels)
print("score on training set : ",rfmodel.score(train_set, train_labels))

print("score on validation set : ",rfmodel.score(val_set, val_labels))