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
submission = pd.read_csv('../input/sample_submission.csv')

submission.head()
train = pd.read_json('../input/train.json')

train.head()
train.describe()
train.columns
train.isnull().sum(axis=0)
del train['id']
cuisines = set()

for cuisine in train['cuisine']:

    cuisines.add(cuisine)

cuisines = list(cuisines)

print(cuisines)
ingredients = set()

for ingredient in train['ingredients']:

    for i in ingredient:

        ingredients.add(i)

for ingredient in test['ingredients']:

    for i in ingredient:

        ingredients.add(i)

ingredients = list(ingredients)
print(len(ingredients))
train.shape
test = pd.read_json('../input/test.json')

print(test.shape)

test.head()

ingredients

def one_hot_vector(unique, data):

    X_data = [0]*len(data)

    print(len(data))

    x = np.zeros(len(unique))

    for i in range(len(data)):

        for ing in data[i]:

            x[unique.index(ing)] = 1

        X_data[i] = x

    return X_data        
x_data = one_hot_vector(ingredients, train['ingredients'])
def one_hot_vector_y(unique, data):

    X_data = [0]*len(data)

    print(len(data))

    x = np.zeros(len(unique))

    for i in range(len(data)):

        x[unique.index(data[i])] = 1

        X_data[i] = x

    return X_data    



y_data = one_hot_vector_y(cuisines, train['cuisine'])
assert(len(x_data) == len(train['ingredients']))

assert (len(x_data[0]) == len(ingredients))

assert(len(y_data) == len(train['ingredients']))

assert(len(y_data[0]) == len(cuisines))
x_test_data = one_hot_vector(ingredients, test['ingredients'])
assert(len(x_test_data) == len(test['ingredients']))

assert (len(x_test_data[0]) == len(ingredients))
from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []

test_accuracy = []

for n_neighbors in range(1, 11):

    # build the model

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn.fit(x_data, y_data)

    # record training set accuracy

    training_accuracy.append(knn.score(x_data, y_data))
import matplotlib.pyplot as plt

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")

plt.ylabel("Accuracy")

plt.xlabel("n_neighbors")

plt.legend()