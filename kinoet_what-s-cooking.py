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
from sklearn.feature_extraction.text import TfidfVectorizer

# data preprocessing
train_data = pd.read_json('../input/train.json')
test_data = pd.read_json('../input/test.json')

train_data['ingredients'] = train_data['ingredients'].apply(lambda list: ','.join(list).lower())
test_data['ingredients'] = test_data['ingredients'].apply(lambda list: ','.join(list).lower())

vectorizer = TfidfVectorizer(binary = True)
train_X = vectorizer.fit_transform(train_data['ingredients'])
test_X = vectorizer.transform(test_data['ingredients'])
idxtuple = train_X.nonzero()
for i in range(16):
    row = idxtuple[0][i]
    col = idxtuple[1][i]
    print('Recipe {0}: {1} = {2}'.format(row, vectorizer.get_feature_names()[col], train_X[row, col]))
# Naive Bayes, score = 0.68634
'''
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(train_X, train_data['cuisine'])

test_pred = clf.predict(test_X)
print(test_pred)
answer = pd.DataFrame({'id': test_data['id'], 'cuisine': test_pred})
answer.to_csv('answer.csv', index = False)
'''
# SVC with panelty = 100, socre = 0.71721, but take too long time
'''
from sklearn.svm import SVC

clf = SVC(C = 100).fit(train_X, train_data['cuisine'])

test_pred = clf.predict(test_X)
print(test_pred)
answer = pd.DataFrame({'id': test_data['id'], 'cuisine': test_pred})
answer.to_csv('answer.csv', index = False)
'''
# LinearSVC with OvR classifier, score = 0.78761 and very fast
'''
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_X, train_data['cuisine'])

test_pred = clf.predict(test_X)
print(test_pred)
answer = pd.DataFrame({'id': test_data['id'], 'cuisine': test_pred})
answer.to_csv('answer.csv', index = False)
'''
# LinearSVC with OvO classifier, score = 0.78821
'''
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
clf = OneVsOneClassifier(LinearSVC(random_state=0)).fit(train_X, train_data['cuisine'])

test_pred = clf.predict(test_X)
print(test_pred)
answer = pd.DataFrame({'id': test_data['id'], 'cuisine': test_pred})
answer.to_csv('answer.csv', index = False)
'''
# My toy model(?)
train_data = pd.read_json('../input/train.json')
test_data = pd.read_json('../input/test.json')

cuisine = train_data['cuisine'].value_counts()
ingredients = {}
for idx, row in train_data.iterrows():
    for i in range(len(row['ingredients'])):
        ingr = row['ingredients'][i].lower()
        if row['cuisine'] in ingredients:
            if ingr in ingredients[row['cuisine']]:
                ingredients[row['cuisine']][ingr] += 1
            else:
                ingredients[row['cuisine']][ingr] = 1
        else:
            ingredients[row['cuisine']] = {}
            ingredients[row['cuisine']][ingr] = 1

ingredients = pd.DataFrame(ingredients)
ingredients = ingredients.fillna(0)
# term frequency
ingredients = ingredients.apply(lambda x: x.apply(lambda y: y / cuisine[x.name]))

# unit-vectorize
def unit_vec(vec):
    len = (vec.apply(lambda x: x * x).sum()) ** 0.5
    return vec.apply(lambda x: x / len)
ingredients = ingredients.apply(unit_vec, axis = 1)
# get recipe vector on training data
train_data_vector = []
for idx, row in train_data.iterrows():
    train_data_vector.append({
        'id': row['id'],
        'cuisine': row['cuisine'],
        'cuisine_vector': ingredients.loc[row['ingredients']].sum()
    })

# Evaluate
total = len(train_data_vector)
correct = 0
for i in range(total):
    if train_data_vector[i]['cuisine_vector'].idxmax() == train_data_vector[i]['cuisine']:
        correct += 1

print('Result on training data')
print(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total}')
# get recipe vector on testing data and predict(?)
testing_data_vector = []
for idx, row in test_data.iterrows():
    ingredients_known = [igdt for igdt in row['ingredients'] if igdt in ingredients.index]
    vector = ingredients.loc[ingredients_known].sum()
    cuisine = vector.idxmax()
    testing_data_vector.append({
        'id': row['id'],
        'cuisine_vector': vector,
        'cuisine': cuisine
    })
answer = pd.DataFrame(testing_data_vector)[['id','cuisine']]
answer.to_csv('answer.csv', index = False)