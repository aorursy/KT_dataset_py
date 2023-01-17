# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# It will takes about 1 ~ 2 minutes (depends on CPU)
train_data = np.genfromtxt('../input/train.csv', delimiter=',',
                  skip_header=1).astype(np.dtype('uint8'))
X_train = train_data[:,1:]
y_train = train_data[:,:1]

X_test = np.genfromtxt('../input/test.csv', delimiter=',',
                  skip_header=1).astype(np.dtype('uint8'))
m_train = X_train.shape[0]
m_test = X_test.shape[0]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
np.random.seed(0);
indices = list(np.random.randint(m_train, size=9))
for i in range(9):
    plt.subplot(3,3,i + 1)
    plt.imshow(X_train[indices[i]].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Index {} Class {}".format(indices[i], y_train[indices[i]]))
    plt.tight_layout()
def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector1 - vector2, 2)))
def absolute_distance(vector1, vector2):
    return np.sum(np.absolute(vector1 - vector2))
import operator
def get_neighbours(X_train, test_instance, k):
    distances = []
    neighbors = []
    for i in range(0, X_train.shape[0]):
        dist = euclidean_distance(X_train[i], test_instance)
        distances.append((i, dist))
    distances.sort(key=operator.itemgetter(1))
    for x in range(k):
        # print(distances[x])
        neighbors.append(distances[x][0])
    return neighbors
def predictkNNClass(output, y_train):
    classVotes = {}
    for i in range(len(output)):
    # print(output[i], y_train[output[i]])
        if y_train[output[i]][0] in classVotes:
            classVotes[y_train[output[i]][0]] += 1
        else:
            classVotes[y_train[output[i]][0]] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    # print(sortedVotes)
    return sortedVotes[0][0]
instance_num = 0
k = 9
plt.imshow(X_test[instance_num].reshape(28,28), cmap='gray', interpolation='none')
instance_neighbours = get_neighbours(X_train, X_test[instance_num], 9)
indices = instance_neighbours
for i in range(9):
    plt.subplot(3,3,i + 1)
    plt.imshow(X_train[indices[i]].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Index {} Class {}".format(indices[i], y_train[indices[i]]))
    plt.tight_layout()
predictkNNClass(instance_neighbours, y_train)
import csv
submit = pd.DataFrame(columns=('ImageId', 'Label'))
for i in range(5):  # change 5 to X_test.shapep[0] will takes a long long long ... TIME!
    neighbours = get_neighbours(X_train, X_test[i], 20)
    label = predictkNNClass(neighbours, y_train)
    submit.loc[i]={'ImageId': i + 1,'Label': label}
submit
submit.to_csv('csv_to_submit.csv', index = False)