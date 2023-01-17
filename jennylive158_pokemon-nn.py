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
# importing libraries

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras

import math

from sklearn.metrics import mean_squared_error
# getting data

dataset = pd.read_csv('../input/Pokemon.csv')



# drop rows with missing 'Legendary' values

dataset.dropna(axis=0, subset=['Legendary'], inplace=True)



dataset.head()
# separate target from predictors

y = dataset.Legendary

X = dataset.drop('Legendary', axis=1)

X = X.drop('#', axis=1)
X.head()
# get list of categorical variables

s = (X.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
# drop unconsidered variables

X.drop('Name', axis=1, inplace=True)



object_cols.remove('Name')
# dealing with missing data



# get names of columns with missing values

cols_with_missing = [col for col in X.columns

                    if X[col].isnull().any()]



print(cols_with_missing)
# decided what to do with missing values

# mark them as 0

X.fillna(' ', inplace=True)



# change all 'object' dtypes to strings

str_all = np.vectorize(str)

str_all(X[object_cols])
# one-hot encoding categorical variables

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols = pd.DataFrame(OH_encoder.fit_transform(X[object_cols]))



# # one-hot encoding removed index, put it back

# OH_cols.index = X.index

# print(OH_cols.index)



# remove categorical columns, replaced with one-hot encoding

X.drop(object_cols, axis=1, inplace=True)

X = pd.concat([X, OH_cols], axis=1)



print("one-hot encoding done")
# scaling the data

sc = MinMaxScaler(feature_range=(0, 1))

X_scaled = sc.fit_transform(X)
train_X, test_X, train_y, test_y = train_test_split(X_scaled, y, train_size=0.8, test_size=0.2, random_state=1)
train_X.shape
num_features = train_X.shape[1]

num_outputs = 2



model = keras.Sequential([

    keras.layers.Dense(num_features, activation=tf.nn.relu),

    keras.layers.Dense(num_outputs, activation=tf.nn.softmax)

])
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
train_y = train_y.values

test_y = test_y.values
model.fit(train_X, train_y, epochs=5)
test_loss, test_acc = model.evaluate(test_X, test_y)



print('Test accuracy:', test_acc)
print(test_X[0])

test_X[0].shape
def pre_process_data(X):

    s = (X.dtypes == 'object')

    object_cols = list(s[s].index)

    

    # drop unconsidered variables

    X.drop('Name', axis=1, inplace=True)

    object_cols.remove('Name')



    # decided what to do with missing values

    # mark them as 0

    X.fillna(' ', inplace=True)



    # change all 'object' dtypes to strings

    str_all = np.vectorize(str)

    str_all(X[object_cols])



    # one-hot encoding categorical variables

    OH_cols = pd.DataFrame(OH_encoder.transform(X[object_cols]))



    # one-hot encoding removed index, put it back

    OH_cols.index = X.index



    # remove categorical columns, replaced with one-hot encoding

    X.drop(object_cols, axis=1, inplace=True)

    X = pd.concat([X, OH_cols], axis=1)



    X_scaled = sc.transform(X)

    

    return X_scaled
# make a single prediction on new pokemon(s)

new_df = pd.DataFrame({"Name":["tylerino"], "Type 1":["Grass"], "Type 2":[None], "Total":[365], "HP":[50], "Attack":[63], 

                               "Defense":[60], "Sp. Atk":[62], "Sp. Def":[60], "Speed":[50], "Generation":[2]})

new_names = new_df.Name



new_processed_df = pre_process_data(new_df)
print(new_processed_df[0])
new_processed_data = (np.expand_dims(new_processed_df[0],0))



print(new_processed_data.shape)
new_predictions = model.predict(new_processed_data)

print(new_predictions)
prediction_result = np.argmax(new_predictions[0])



if prediction_result == 0:

    print(new_names.values[0], "is not a Legendary Pokemon, what a noob!")

else:

    print(new_names.values[0], "is a Legendary Pokemon, amazing!")