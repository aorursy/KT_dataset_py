# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Libraries import
import pandas as pd
import numpy as np
import csv as csv
import json
import re
import random #Used to randomly choose ingredients

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
with open('../input/train.json', 'r') as f:
    train = json.load(f)
train_raw_df = pd.DataFrame(train)

with open('../input/test.json', 'r') as f:
    test = json.load(f)
test_raw_df = pd.DataFrame(test)
# Remove numbers and only keep words
# substitute the matched pattern
# update the ingredients
def sub_match(pattern, sub_pattern, ingredients):
    for i in ingredients.index.values:
        for j in range(len(ingredients[i])):
            ingredients[i][j] = re.sub(pattern, sub_pattern, ingredients[i][j].strip())
            ingredients[i][j] = ingredients[i][j].strip()
    re.purge()
    return ingredients

#remove units
p0 = re.compile(r'\s*(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\s*[^a-z]')
train_raw_df['ingredients'] = sub_match(p0, ' ', train_raw_df['ingredients'])
# remove digits
p1 = re.compile(r'\d+')
train_raw_df['ingredients'] = sub_match(p1, ' ', train_raw_df['ingredients'])
# remove non-letter characters
p2 = re.compile('[^\w]')
train_raw_df['ingredients'] = sub_match(p2, ' ', train_raw_df['ingredients'])

y_train = train_raw_df['cuisine'].values
train_ingredients = train_raw_df['ingredients'].values
train_ingredients_update = list()
for item in train_ingredients:
    item = [x.lower().replace(' ', '+') for x in item]
    train_ingredients_update.append(item)
X_train = [' '.join(x) for x in train_ingredients_update]
# Create the dataframe for creating new recipes
food_df = pd.DataFrame({'cuisine':y_train
              ,'ingredients':train_ingredients_update})
# the randomly picked function
def random_generate_recipe(raw_df, food_type, num_ingredients):
    if food_type not in raw_df['cuisine'].values:
        print('Food type is not existing here')
    food_ingredients_lst = list()
    [food_ingredients_lst.extend(recipe) for recipe in raw_df[raw_df['cuisine'] == food_type]['ingredients'].values] 
    i = 0
    new_recipe, tmp = list(), list()
    while i < num_ingredients:
        item = random.choice(food_ingredients_lst)
        if item not in tmp:
            tmp.append(item)
            new_recipe.append(item.replace('+', ' '))
            i+=1
        else:
            continue
    recipt_str = ', '.join(new_recipe)
    print('The new recipte for %s can be: %s' %(food_type, recipt_str))
    return new_recipe
#Say you want some chinese food and you want to only have 10 ingredients in it
random_generate_recipe(food_df, 'chinese', 10)
#Say you want some indian food and you want to only have 12 ingredients in it
random_generate_recipe(food_df, 'indian', 12)
#Say you want some french food and you want to only have 8 ingredients in it
random_generate_recipe(food_df, 'french', 12)
