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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
with open('../input/train.json') as f:
    train = json.load(f)
ie = inflect.engine()
def get_ingredients_from_list_of_ingredients(ingredients):
    ingredients = ' '.join(s.strip() for s in ingredients)
    ingredients = ingredients.split()
    single_ingredients = [ie.singular_noun(s) for s in ingredients]
    ingredients = [single_ingredients[i] if single_ingredients[i] else ingredients[i] for i in range(len(ingredients))]
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
from sklearn.preprocessing import OneHotEncoder
train_y_oh = scsp.csr_matrix(np.eye(len(cuisines_to_index))[train_y])
alpha = 10.0
prob_of_ingredient_given_cuisine = np.multiply(train_y_oh.transpose(), train_X)
prob_of_ingredient_given_cuisine = alpha + prob_of_ingredient_given_cuisine.toarray()
prob_of_ingredient_given_cuisine = prob_of_ingredient_given_cuisine.transpose() / prob_of_ingredient_given_cuisine.sum(1)
prob_of_ingredient_given_cuisine = prob_of_ingredient_given_cuisine.transpose()
cuisine_counter = Counter(train_y)
dish_count = len(train_y)
prob_cuisine = np.array([cuisine_counter[cuisine] / dish_count for cuisine in range(len(cuisine_counter.keys()))])
def get_prediction(X, prob_of_ingredient_given_cuisine, prob_of_cuisine):
    log_prob_of_ingredient_given_cuisine = np.log(prob_of_ingredient_given_cuisine)
    log_prob_cuisine = np.log(prob_cuisine)
    prediction = log_prob_cuisine + X.dot(log_prob_of_ingredient_given_cuisine.transpose())
    return prediction.argmax(1)
prediction = get_prediction(train_X, prob_of_ingredient_given_cuisine, prob_cuisine)
actual = np.array([cuisines_to_index[dish['cuisine']] for dish in train])
miss_classification_count = np.count_nonzero(prediction != actual)
1 - miss_classification_count / len(actual)
truth_vs_prediction_matrix = np.zeros((len(cuisines_to_index), len(cuisines_to_index)), dtype=np.int32)
for i in range(len(prediction)):
    truth_vs_prediction_matrix[actual[i]][prediction[i]] += 1
import seaborn
seaborn.heatmap((truth_vs_prediction_matrix) * (1 - np.eye(len(cuisines_to_index))),square=True, xticklabels=cuisines_to_index.keys(), yticklabels=cuisines_to_index.keys())
np.set_printoptions(threshold=np.nan)
with open('../input/test.json') as f:
    test = json.load(f)
test_X, _ = convert_dataset_to_ndarray(test, ingredients_to_index, cuisines_to_index)
test_X.shape
len(ingredients_to_index)
test_X_ = test_X[:,:len(ingredients_to_index.keys())]
test_X_.shape
prediction = get_prediction(test_X_, prob_of_ingredient_given_cuisine, prob_cuisine)
test_ids = [sample['id'] for sample in test]
index_to_cuisine = {v: k for k, v in cuisines_to_index.items()}
with open('prediction.csv', 'w') as f:
    f.write('id,cuisine\n')
    for test_sample_id, predicted_cuisine in zip(test_ids, prediction):
        f.write(str(test_sample_id) + ',' + str(index_to_cuisine[predicted_cuisine]) + '\n')