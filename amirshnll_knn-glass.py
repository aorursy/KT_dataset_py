# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/glass/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/glass/glass.csv', encoding ='latin1')

data.info()
data.head(20000)
y = data['Type'].values

y = y.reshape(-1,1)

x_data = data.drop(['Type'],axis = 1)

print(x_data)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

x.head(20000)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.5,random_state=100)



y_train = y_train.reshape(-1,1)

y_test = y_test.reshape(-1,1)



print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
from sklearn.neighbors import KNeighborsClassifier

K = 1

knn = KNeighborsClassifier(n_neighbors=K)

knn.fit(x_train, y_train.ravel())

print("When K = {} neighnors , KNN test accuracy: {}".format(K, knn.score(x_test, y_test)))

print("When K = {} neighnors , KNN train accuracy: {}".format(K, knn.score(x_train, y_train)))
ran = np.arange(1,30)

train_list = []

test_list = []

for i,each in enumerate(ran):

    knn = KNeighborsClassifier(n_neighbors=each)

    knn.fit(x_train, y_train.ravel())

    test_list.append(knn.score(x_test, y_test))

    train_list.append(knn.score(x_train, y_train))
plt.figure(figsize=[15,10])

plt.plot(ran,test_list,label='Test Score')

plt.plot(ran,train_list,label = 'Train Score')

plt.xlabel('Number of Neighbers')

plt.ylabel('RI/Na/Mg/Al/Si/K/Ca/Ba/Fe')

plt.xticks(ran)

plt.legend()

print("Best test score is {} and K = {}".format(np.max(test_list), test_list.index(np.max(test_list))+1))

print("Best train score is {} and K = {}".format(np.max(train_list), train_list.index(np.max(train_list))+1))