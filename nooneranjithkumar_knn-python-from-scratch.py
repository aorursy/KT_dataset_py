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
data = pd.read_csv('../input/iris-data/Iris.csv')

data.head(5)
Species = list(set(data['Species']))

Specie1 = data[data['Species']==Species[0]]

Specie2 = data[data['Species']==Species[1]]

Specie3 = data[data['Species']==Species[2]]
import matplotlib.pyplot as plt

plt.scatter(Specie1['PetalLengthCm'], Specie1['PetalWidthCm'], label=Species[0])

plt.scatter(Specie2['PetalLengthCm'], Specie2['PetalWidthCm'], label=Species[1])

plt.scatter(Specie3['PetalLengthCm'], Specie3['PetalWidthCm'], label=Species[2])

plt.xlabel('PetalLengthCM')

plt.ylabel('PetalWidthCM')

plt.legend()

plt.title('Different Species Visualization')
req_data = data.iloc[:,1:]

req_data.head(5)
shuffle_index = np.random.permutation(req_data.shape[0])        #shuffling the row index of our dataset

req_data = req_data.iloc[shuffle_index]

req_data.head(5)
train_size = int(req_data.shape[0]*0.7)
train_df = req_data.iloc[:train_size,:] 

test_df = req_data.iloc[train_size:,:]

train = train_df.values

test = test_df.values

y_true = test[:,-1]

print('Train_Shape: ',train_df.shape)

print('Test_Shape: ',test_df.shape)
from math import sqrt

def euclidean_distance(x_test, x_train):

    distance = 0

    for i in range(len(x_test)-1):

        distance += (x_test[i]-x_train[i])**2

    return sqrt(distance)
def get_neighbors(x_test, x_train, num_neighbors):

    distances = []

    data = []

    for i in x_train:

        distances.append(euclidean_distance(x_test,i))

        data.append(i)

    distances = np.array(distances)

    data = np.array(data)

    sort_indexes = distances.argsort()             #argsort() function returns indices by sorting distances data in ascending order

    data = data[sort_indexes]                      #modifying our data based on sorted indices, so that we can get the nearest neightbours

    return data[:num_neighbors]               
def prediction(x_test, x_train, num_neighbors):

    classes = []

    neighbors = get_neighbors(x_test, x_train, num_neighbors)

    for i in neighbors:

        classes.append(i[-1])

    predicted = max(classes, key=classes.count)              #taking the most repeated class

    return predicted
def predict_classifier(x_test):

    classes = []

    neighbors = get_neighbors(x_test, req_data.values, 5)

    for i in neighbors:

        classes.append(i[-1])

    predicted = max(classes, key=classes.count)

    print(predicted)

    return predicted
def accuracy(y_true, y_pred):

    num_correct = 0

    for i in range(len(y_true)):

        if y_true[i]==y_pred[i]:

            num_correct+=1

    accuracy = num_correct/len(y_true)

    return accuracy
y_pred = []

for i in test:

    y_pred.append(prediction(i, train, 5))

y_pred
accuracy = accuracy(y_true, y_pred)
accuracy
test_df.insert(5, 'Predicted_Species', y_pred, False)
test_df.sample(5)