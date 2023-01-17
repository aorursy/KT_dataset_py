# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!ls -la

data_path = '../input/iris-flower-dataset/IRIS.csv'



ori_data = pd.read_csv(data_path)



print(ori_data.info)
species_class = set(ori_data['species'].values)



print(species_class)



def shuffle(x,y):

    indexes = np.arange(len(x))

    np.random.shuffle(indexes)

    return x[indexes],y[indexes]



def distance_l_2(x1,x2):

    x = x1 - x2

    return np.sqrt(np.dot(x.T,x))



def find_k_neighbor(k,x,data,labels):

    distances = []

    for i,x_i in enumerate(data):

        distances.append((i,distance_l_2(x,x_i)))

    distances.sort(key=lambda tup: tup[1])

    neighbors = []

    for i in range(k):

        idx = distances[i][0]

        neighbors.append((data[idx],labels[idx]))

    return neighbors



def find_classes(k,neighbors, labels):

    num = k/3.0

    nums = [0,0,0]

    for neighbor in neighbors:

        data, label = neighbor[0],neighbor[1]

        for i in range(3):

            if label == labels[i]:

                nums[i] += 1

            if nums[i] >= num:

                return label

    



def split_data(x, y, rate = 0.25):

    test_data_len = int(len(x)*rate)

    return x[:test_data_len],x[test_data_len:],y[:test_data_len],y[test_data_len:]







data, labels = ori_data.iloc[:,0:2].values, ori_data['species'].values



# 'Iris-virginica', 'Iris-setosa', 'Iris-versicolor'



c1,c2,c3 = data[:50],data[50:100],data[100:]



plt.scatter(c1[:,0],c1[:,1],c = 'r')

plt.scatter(c2[:,0],c2[:,1],c = 'b')

plt.scatter(c3[:,0],c3[:,1],c = 'y')



plt.show()







data,labels = shuffle(data, labels)

test_data,train_data,test_labels,train_labels = split_data(data,labels)

k = 5

classes = ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']

error = 0

l = len(test_data)

for idx in range(len(test_data)):

    x, label = test_data[idx], test_labels[idx]

    neighbors = find_k_neighbor(k,x,train_data,train_labels)

    pred_label = find_classes(k, neighbors,classes)

    print(x, label,pred_label)

    if label != pred_label:

        error += 1

print('accrate rate is {}'.format(1.0 - error/l*1.0))
