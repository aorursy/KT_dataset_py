# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import matplotlib.pyplot as plt
import inflect
from collections import Counter
import scipy.sparse as scsp
from sklearn.linear_model import LogisticRegression
import itertools

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
with open('../input/train.json') as f:
    train = json.load(f)
ie = inflect.engine()
def get_singular(w):
    s = ie.singular_noun(w)
    if s:
        return s
    else:
        return w
def get_bigrams(s):
    s = s.strip()
    s = s.lower()
    s = s.split()
    s = [get_singular(w) for w in s]
    if len(s) == 1:
        return [(s[0], '')]
    ret = []
    for i in range(1, len(s)):
        ret.append((s[i-1], s[i]))
    return ret
def get_ingredients_from_list_of_ingredients(ingredients):
    ingredients = [get_bigrams(s) for s in ingredients]
    ingredients = itertools.chain.from_iterable(ingredients)
    #ingredients = [bigram for ingredient in ingredients for bigram in ingredient]
    return ingredients
def convert_dataset_to_ndarray(dataset, ingredients_to_index=None, cuisines_to_index=None):
    if ingredients_to_index is None:
        ingredients_to_index = dict()
        cuisines_to_index = dict()
    else:
        ingredients_to_index = dict(ingredients_to_index)
        cuisines_to_index = dict(cuisines_to_index)
    
    dish_ingredient_count = list()
    dish_cuisine = list()
    cuisine_present = True
    for sample in dataset:
        if cuisine_present:
            try:
                if sample['cuisine'] not in cuisines_to_index:
                    cuisines_to_index[sample['cuisine']] = len(cuisines_to_index.keys())
                dish_cuisine.append(cuisines_to_index[sample['cuisine']])
                cuisine_present = True
            except KeyError:
                cuisine_present = False
            
        dish_ingredient_count.append(list())
        ingredients_count = Counter(get_ingredients_from_list_of_ingredients(sample['ingredients']))
        for ingredient in ingredients_count:
            if ingredient not in ingredients_to_index:
                ingredients_to_index[ingredient] = len(ingredients_to_index.keys())
            dish_ingredient_count[-1].append((ingredients_to_index[ingredient], ingredients_count[ingredient]))
    
    data = []
    row_indices = []
    col_indices = []
    cur_row = 0
    for ingredient_count in dish_ingredient_count:
        for ingredient, count in ingredient_count:
            data.append(count)
            row_indices.append(cur_row)
            col_indices.append(ingredient)
        cur_row += 1
    X = scsp.csr_matrix((data, (row_indices, col_indices)))
    if cuisine_present:
        y = np.array(dish_cuisine)
        return X, y, ingredients_to_index, cuisines_to_index
    else:
        return X, ingredients_to_index
train_X, train_y, ingredients_to_index, cuisines_to_index = convert_dataset_to_ndarray(train)
train_X = train_X.astype(np.float64)
train_X.shape
ingredients_to_index
index_to_cuisine = {v: k for k, v in cuisines_to_index.items()}
class_weights = [1.0] * len(cuisines_to_index.keys())

class_weights[cuisines_to_index['italian']] = 1.0
class_weights[cuisines_to_index['french']] = 2.0
class_weights[cuisines_to_index['spanish']] = 2.0
class_weights[cuisines_to_index['vietnamese']] = 2.0
class_weights[cuisines_to_index['british']] = 2.5
class_weights[cuisines_to_index['irish']] = 2.0
class_weights[cuisines_to_index['russian']] = 2.0
class_weights[cuisines_to_index['greek']] = 2.0
class_weights = dict((c,i) for c,i in enumerate(class_weights))
class_weights
clf = LogisticRegression(verbose=5)
clf.fit(train_X, train_y)
clf.score(train_X, train_y)
prediction = clf.predict(train_X)
actual = np.array([cuisines_to_index[dish['cuisine']] for dish in train])
truth_vs_prediction_matrix = np.zeros((len(cuisines_to_index), len(cuisines_to_index)), dtype=np.int32)
for i in range(len(prediction)):
    truth_vs_prediction_matrix[actual[i]][prediction[i]] += 1
import seaborn
cuisine_counter = Counter(train_y)
truth_vs_prediction_matrix = (truth_vs_prediction_matrix.transpose() / np.array(list(cuisine_counter.values()))).transpose()
seaborn.heatmap((truth_vs_prediction_matrix) * (1 - np.eye(len(cuisines_to_index))),square=True, vmin=0.0, xticklabels=cuisines_to_index.keys(), yticklabels=cuisines_to_index.keys())
with open('../input/test.json') as f:
    test = json.load(f)
test_X, _ = convert_dataset_to_ndarray(test, ingredients_to_index, cuisines_to_index)
test_X.shape
len(ingredients_to_index)
test_X_ = test_X[:,:len(ingredients_to_index.keys())]
test_X_.shape
prediction = clf.predict(test_X_)
test_ids = [sample['id'] for sample in test]
with open('prediction.csv', 'w') as f:
    f.write('id,cuisine\n')
    for test_sample_id, predicted_cuisine in zip(test_ids, prediction):
        f.write(str(test_sample_id) + ',' + str(index_to_cuisine[predicted_cuisine]) + '\n')
