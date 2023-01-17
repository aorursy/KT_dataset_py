# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
iris = load_iris()

y_iris = iris.target

data = iris.data
iris.keys()


data = np.insert(data,4,y_iris,axis = 1)
train,test = train_test_split(data, test_size = 0.3)
from math import sqrt

def euclidean_distance(row1, row2):

    distance = 0

    for i in range(len(row1)-1):

        distance += (row1[i] - row2[i])**2

    return sqrt(distance)

def get_neighbors(train, test_row, num_neighbors):

    distances = list()

    data = []

    for train_row in train:

        dist = euclidean_distance(test_row, train_row)

        distances.append(dist)

        data.append(train_row)

    distance = np.array(distances)

    data = np.array(data)

    index_dist = distance.argsort()

    data = data[index_dist]

    neighbors = data[:num_neighbors]

    return neighbors

def predict_classification(train, test_row, num_neighbors):

    neighbors = get_neighbors(train, test_row, num_neighbors)

    classes = []

    for row in neighbors:

        classes.append(row[-1])

    prediction = max(classes, key=classes.count)

    return prediction

prediction = predict_classification(train,test[-1],4)

print('Expected %d, Got %d.' % (data[0][-1], prediction))
y_pred = []

y_true = test[:,-1]

for i in test:

    prediction = predict_classification(train,i,10)

    y_pred.append(prediction)
def Evaluate(y_true,y_pred):

    n_correct = 0

    for i in range(len(y_true)):

        if y_true[i] == y_pred[i]:

            n_correct += 1

    acc = n_correct/len(y_true)

    return acc

Evaluate(y_true, y_pred)