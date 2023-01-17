
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
##### EXPLORE #########==================
# data exploring and basic libraries
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from collections import deque as dq

# NLP preprocessing
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize as TK
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

##### MODELING ######===================
# from time import time
# train test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as LE

# deep learning
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, PReLU
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

# Pretty display for notebooks
from IPython.display import display # Allows the use of display() for DataFrames
%matplotlib inline
# load the data - test data
rawdf_te = pd.read_json(path_or_buf='../input/test.json')
# load the data - train data
rawdf_tr = pd.read_json(path_or_buf='../input/train.json')
# cuisine distribution
sns.countplot(y='cuisine', data=rawdf_tr, palette ='Set3')

# number of recipes for each cuisines
print('Weight\t Recipe\t Cuisine\n')
for _ in (Counter(rawdf_tr['cuisine']).most_common()):print(round(_[1]/rawdf_tr.cuisine.count()*100, 2),'%\t',_[1],'\t', _[0])
# change id column type to string
rawdf_tr = rawdf_tr.set_index('id')
rawdf_te = rawdf_te.set_index('id')

# Total number of recipes
print('Total of %d recipes\n'% len(rawdf_tr))

# total number of UNIQUE cuisines
print('Total of %d types of cuisines including %s\n' % \
      (len(rawdf_tr['cuisine'].unique()), rawdf_tr['cuisine'].unique().tolist()))
                                          
# UNIQUE # ingredients set - ingredients_set()
# training ingredient list
ingredients_list_tr = []
for _ in rawdf_tr['ingredients']:
    ingredients_list_tr.append(_)
# ingredients set - ingredients_set()
ingredients_set_tr = set()
for a in range(len(ingredients_list_tr)):
    for _ in range(len(ingredients_list_tr[a])):
        ingredients_set_tr.add(ingredients_list_tr[a][_])
print("Total of %d unique ingredients\n" % len(ingredients_set_tr))

# total ingredients list (with repition) occurred in the train data
total_ingredients_list_tr = []
for i in range(len(ingredients_list_tr)):
    for j in range(len(ingredients_list_tr[i])):
        total_ingredients_list_tr.append(ingredients_list_tr[i][j])
print("Most common ingredients used:\n")
for _ in range(len(Counter(total_ingredients_list_tr).most_common(11))):
    print(Counter(total_ingredients_list_tr).most_common(11)[_])
# copy the series from the dataframe
ingredients_tr = rawdf_tr['ingredients']
# do the test.json while at it
ingredients_te = rawdf_te['ingredients']

# substitute the matched pattern
def sub_match(pattern, sub_pattern, ingredients):
    for i in ingredients.index.values:
        for j in range(len(ingredients[i])):
            ingredients[i][j] = re.sub(pattern, sub_pattern, ingredients[i][j].strip())
            ingredients[i][j] = ingredients[i][j].strip()
    re.purge()
    return ingredients

def regex_sub_match(series):
    # remove all units
    p0 = re.compile(r'\s*(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\s*[^a-z]')
    series = sub_match(p0, ' ', series)
    # remove all digits
    p1 = re.compile(r'\d+')
    series = sub_match(p1, ' ', series)
    # remove all the non-letter characters
    p2 = re.compile('[^\w]')
    series = sub_match(p2, ' ', series)
    return series
# regex train data
ingredients_tr = regex_sub_match(ingredients_tr)
# declare instance from WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# remove all the words that are not nouns -- keep the essential ingredients
def lemma(series):
    for i in series.index.values:
        for j in range(len(series[i])):
            # get rid of all extra spaces
            series[i][j] = series[i][j].strip()
            # Tokenize a string to split off punctuation other than periods
            token = TK(series[i][j])
            # set all the plural nouns into singular nouns
            for k in range(len(token)):
                token[k] = lemmatizer.lemmatize(token[k])
            token = ' '.join(token)
            # write them back
            series[i][j] = token
    return series
# lemmatize the train data
ingredients_tr = lemma(ingredients_tr)
# copy back to the dataframe
rawdf_tr['ingredients_lemma'] = ingredients_tr
rawdf_tr['ingredients_lemma_string'] = [' '.join(_).strip() for _ in rawdf_tr['ingredients_lemma']]
# basically train_test_split customized to input cuisine name, outputs are 2 lists of indicies for train and test for the cuisine
def tt_split(cuisine):
    cuisine_population = rawdf_tr.loc[(rawdf_tr['cuisine'] == cuisine)].index.values
    train, test = train_test_split(cuisine_population, test_size=0.15, random_state=0)
    train = train.tolist()
    test = test.tolist()
    return train, test

cuisine_list = rawdf_tr['cuisine'].unique().tolist()
# split the training data into 85-15
ix_train = [] # 85% for training (and validation)
ix_test = [] # 15% for hold-out test
for _ in cuisine_list:
    temp_train, temp_test = tt_split(_)
    ix_train += temp_train
    ix_test += temp_test

# DataFrame for training and validation
traindf = rawdf_tr[['cuisine', 'ingredients_lemma_string']].loc[ix_train].reset_index(drop=True)
print(traindf.shape)
testdf = rawdf_tr[['cuisine', 'ingredients_lemma_string']].loc[ix_test].reset_index(drop=True)
print(testdf.shape)
    
# 85% for training and validation ===================
# X_train
X_train_ls = traindf['ingredients_lemma_string']
vectorizertr = TfidfVectorizer(stop_words='english', analyzer="word", max_df=0.65, min_df=2, binary=True)
X_train = vectorizertr.fit_transform(X_train_ls)

# y_train
y_train = traindf['cuisine']
# for xgboost the labels need to be labeled with encoder
le = LE()
y_train_ec = le.fit_transform(y_train)
# for deep learning
y_train_1h = pd.get_dummies(y_train_ec)

# save the 15% data for hold-out test ===============
# X_pred
X_pred_ls = testdf['ingredients_lemma_string']
vectorizerts = TfidfVectorizer(stop_words='english')
X_pred = vectorizertr.transform(X_pred_ls)

# y_true
y_true = testdf['cuisine']
y_true_ec = le.fit_transform(y_true)
# for deep learning
y_true_1h = pd.get_dummies(y_true_ec)
model = Sequential()
model.add(Dense(2161, input_shape=(2161,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='softsign'))
model.add(Dropout(0.58))
model.add(Dense(256, activation='softsign'))
model.add(Dropout(0.67))
model.add(Dense(20, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 20

early_stopping = EarlyStopping(monitor='val_loss', patience=1)

model.fit(X_train,y_train_1h,
          batch_size=30,
          epochs=epochs,
          verbose=2,
          callbacks=[early_stopping],
          validation_data=(X_pred, y_true_1h),
          shuffle=True)
# regex test.json data
ingredients_te = regex_sub_match(ingredients_te)
# lemmatize test.json
ingredients_te = lemma(ingredients_te)
# do the same for the test.json dataset
rawdf_te['ingredients_lemma'] = ingredients_te
rawdf_te['ingredients_lemma_string'] = [' '.join(_).strip() for _ in rawdf_te['ingredients_lemma']]
testdf = rawdf_te[['ingredients_lemma_string']]
print(testdf.shape)
testdf.head(n=2)
# predicting =================
# X_pred
X_pred_ls = testdf['ingredients_lemma_string']
vectorizerts = TfidfVectorizer(stop_words='english')
X_pred = vectorizertr.transform(X_pred_ls)

# y_true
y_pred_nn = le.inverse_transform(model.predict_classes(X_pred))
testdf['cuisine'] = y_pred_nn
d = pd.DataFrame(data=testdf['cuisine'], index=testdf.index).sort_index().reset_index().to_csv('submission_nn.csv', index=False)
