# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
with open('../input/train.json') as f:
    train = json.load(f)
from collections import Counter
cuisine_counter = Counter([sample['cuisine'] for sample in train])
possible_ingredients = set()
total_ingredient_count = 0
for sample in train:
    total_ingredient_count += len(set(' '.join(sample['ingredients']).lower().split()))
    possible_ingredients.update((' '.join(sample['ingredients'])).lower().split())
possible_ingredients = sorted(list(possible_ingredients))
total_ingredient_count / len(possible_ingredients)
len(possible_ingredients)
ingredients_to_index = dict([(ingredient, i)  for (i, ingredient) in enumerate(possible_ingredients)])
cuisines = sorted(list(cuisine_counter.keys()))
cuisines_to_index = dict([(cuisine, i)  for (i, cuisine) in enumerate(cuisines)])
prob_of_ingredient_given_cuisine = np.ones((len(cuisines), len(possible_ingredients)), dtype=np.float)
for sample in train:
    cuisine = cuisines_to_index[sample['cuisine']]
    for ingredient in set((' '.join(sample['ingredients'])).lower().split()):
        ingredient = ingredients_to_index[ingredient]
        prob_of_ingredient_given_cuisine[cuisine][ingredient] += 1
for cuisine in cuisines:
    dish_count = cuisine_counter[cuisine]
    cuisine = cuisines_to_index[cuisine]
    prob_of_ingredient_given_cuisine[cuisine] /= (dish_count + 2)
dish_count = sum(cuisine_counter.values())
prob_cuisine = [cuisine_counter[cuisine] / dish_count for cuisine in cuisines]
def get_prediction(data, prob_of_ingredient_given_cuisine, prob_of_cuisine):
    prob_of_not_ingredient_given_cuisine = 1 - prob_of_ingredient_given_cuisine
    log_prob_of_ingredient_given_cuisine = np.log(prob_of_ingredient_given_cuisine)
    log_prob_of_not_ingredient_given_cuisine = np.log(prob_of_not_ingredient_given_cuisine)
    log_prob_cuisine = np.log(prob_cuisine)
    prediction = []
    for dish in data:
        log_prior = log_prob_cuisine
        ingredients_present = np.zeros((len(possible_ingredients),), np.int8)
        for ingredient in set((' '.join(dish['ingredients'])).lower().split()):
            if ingredient in possible_ingredients:
                ingredients_present[ingredients_to_index[ingredient]] = 1
        log_likelihood = np.zeros_like(log_prior)
        for cuisine in range(len(cuisines)):
            log_likelihood[cuisine] += np.sum(ingredients_present * log_prob_of_ingredient_given_cuisine[cuisine] 
                                          + (1 - ingredients_present) * log_prob_of_not_ingredient_given_cuisine[cuisine])
        predicted_cuisine = np.argmax(log_prior + log_likelihood)
        prediction.append((dish['id'], predicted_cuisine))
    return prediction
prediction = get_prediction(train, prob_of_ingredient_given_cuisine, prob_cuisine)
actual = [dish['cuisine'] for dish in train]
miss_classification_count = 0
for i in range(len(prediction)):
    if prediction[i][1] != cuisines_to_index[actual[i]]:
        miss_classification_count += 1
miss_classification_count
1 - miss_classification_count / len(prediction)
with open('../input/test.json') as f:
    test = json.load(f)
prediction = get_prediction(test, prob_of_ingredient_given_cuisine, prob_cuisine)
with open('prediction.csv', 'w') as f:
    f.write('id,cuisine\n')
    for test_sample_id, predicted_cuisine in prediction:
        f.write(str(test_sample_id) + ',' + str(cuisines[predicted_cuisine]) + '\n')