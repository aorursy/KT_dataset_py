# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import operator
df = pd.read_csv('/kaggle/input/iris/Iris.csv')
df.head()
def euclidian_distance(row1, row2, length):
    '''
    Caculate the euclidian distance between rows
    '''
    distance = 0
    
    for x in range(length):
        distance += np.square(row1[x] - row2[x])
       
    return np.sqrt(distance)
def get_neighbors(dataset, sorted_distances, k):
    '''
    Get the closest neighbors in the range of k elements
    '''
    neighbors = []

    for x in range(k):
        neighbors.append(sorted_distances[x][0])
        
    return neighbors
def get_sorted_distances(dataset, row):
    '''
    Get sorted distance between the row and the dataset
    
    '''
    distances = {}
    
    for x in range(len(dataset)):
        dist = euclidian_distance(row, dataset.iloc[x], row.shape[1])
        distances[x] = dist[0]
      
    sorted_distances = sorted(distances.items(), key=operator.itemgetter(1))
    
    return sorted_distances
def get_sorted_neighbourhood(dataset, neighbors):
    '''
    Get the neighbor that has the most votes
    '''
    neighbourhood = {}
    
    for x in range(len(neighbors)):
        response = dataset.iloc[neighbors[x]][-1]
 
        if response in neighbourhood:
            neighbourhood[response] += 1
        else:
            neighbourhood[response] = 1
            
    sorted_neighbourhood = sorted(neighbourhood.items(), key=operator.itemgetter(1), reverse=True)
    
    return sorted_neighbourhood
def knn(dataset, testInstance, k):
    '''
    Implementation of k-nearest neighbors algorithm
    '''
    
    sorted_distances = get_sorted_distances(dataset, testInstance)
   
    neighbors = get_neighbors(dataset, sorted_distances, k)

    sorted_neighbourhood = get_sorted_neighbourhood(dataset, neighbors)
    
    neighbors.insert(0, sorted_neighbourhood[0][0])
    
    return neighbors
species = dict(zip(list(train_x['Species'].unique()), ([1, 2, 3])))
print(species)
categories = { v:k for (k,v) in species.items() }
print(categories)
df['Species'].replace(species, inplace=True)
df['Species'].unique()
iris = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']]
iris.head()
train = []

for i in range(len(iris)):
    row = pd.DataFrame([list(iris.iloc[i].to_numpy()[0:-1])])
    train.append(knn(iris, row, 3))
train = np.array(train)
setosa_x, setosa_y, setosa_z = train[train[:,0] == 1][:,1:][:,0], train[train[:,0] == 1][:,1:][:,1], train[train[:,0] == 1][:,1:][:,2]
versicolor_x, versicolor_y, versicolor_z = train[train[:,0] == 2][:,1:][:,0], train[train[:,0] == 2][:,1:][:,1], train[train[:,0] == 2][:,1:][:,2]
virginica_x, virginica_y, virginica_z = train[train[:,0] == 3][:,1:][:,0], train[train[:,0] == 3][:,1:][:,1], train[train[:,0] == 3][:,1:][:,2]
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(setosa_x, setosa_y, setosa_z, c='r', marker='o')
ax.scatter(versicolor_x, versicolor_y, versicolor_z, c='b', marker='o')
ax.scatter(virginica_x, virginica_y, virginica_z, c='g', marker='o')

ax.set_xlabel('X Neighbor')
ax.set_ylabel('Y Neighbor')
ax.set_zlabel('Z Neighbor')

plt.show()
def predict(sepal_length, sepal_width, petal_length, petal_width):
    row = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]])
    result = knn(iris, row, 3)
    neighbors = result[1:]
    category = categories[result[0]]
    return category, neighbors
sepal_length = 5.0
sepal_width = 3.0

petal_length = 2.0
petal_width = 4.0

category, neighbors = predict(sepal_length, sepal_width, petal_length, petal_width)
print(category)
print(neighbors)