# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import json
 
# load data from input file 
recipe_data = json.loads(open('../input/train.json').read())
# empty set for ingredients
unique_ing = set()
recipe_id = []
cuisine = []
# find out how many unique ingredients exist

for recipe in recipe_data:
    recipe_id.append(recipe['id'])
    cuisine.append(recipe['cuisine'])
    
    # add items to set with faster unity operator
    unique_ing |= set(recipe['ingredients'])

# check length
len(cuisine)
# add to use the unique set as column names
#unique_ing.add('cuisine')
colnames = list(unique_ing)

# create empty dataframe with the number of all recipes and the
data = pd.DataFrame(0, index=recipe_id, columns=colnames)
data.head()
# replace cuisine column with actual cuisine data
data['cuisine'] = cuisine
# fill  in 1 for every matching ingredient
# go over info again
for recipe in recipe_data:
    index = recipe['id']
    ingredients =recipe['ingredients']
    for ingredient in ingredients:
        data.at[index, ingredient] = 1
# factorize the cuisines
y, label = pd.factorize(data['cuisine'])
import keras

# prepare data to be used in model training
X_train = data[colnames].values.astype(float)

# transform data to categoricals
y_train = keras.utils.to_categorical(y, num_classes=20)
# try it with keras und sequential model
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Activation, BatchNormalization, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU

# two dense layers
model = Sequential()
model.add(Dropout(0.3))
model.add(Dense(512, input_dim=6714, activation='linear'))
model.add(LeakyReLU(alpha=.02)) 
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))

model.add(Dense(20, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
model.fit(X_train, y_train,
          epochs=25,
          batch_size=250)
          #validation_split=0.2)
# load test data
test_data = json.loads(open('../input/test.json').read())
# get recipe id's
test_recipe_id = []
# find out how many unique ingredients exist

for recipe in test_data:
    test_recipe_id.append(recipe['id'])

# build empty dataframe with indices
test_df = pd.DataFrame(0, index=test_recipe_id, columns=colnames)
# create dict to check for known ingredients
# cannot predict anything with previously unknown ingredients
ingr_checker = dict.fromkeys(colnames)
# fill dataframe with ingredients
for recipe in test_data:
    index = recipe['id']
    ingredients = recipe['ingredients']
    for ingredient in ingredients:
        # check if ingredient is known
        if ingredient in ingr_checker:
            test_df.at[index, ingredient] = 1
# prepare test data for prediction with trained model
X_test = test_df[colnames].values.astype(float)
prediction = model.predict(X_test)
# get class with highest prob
prediction_classes =  prediction.argmax(axis=-1)
# get the original names for cuisines
label_names = label[prediction_classes]
# build dataframe that will then be used to save output
df_output = pd.DataFrame({'id' : test_recipe_id, 'cuisine' : label_names})

df_output.head()
# save df to csv
df_output.to_csv("output.csv", header=True, index=False)