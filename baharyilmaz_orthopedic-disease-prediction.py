import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# read dataset

data = pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv')
data.tail()
data.info()
data.describe()
# green: Normal , red: Hernia , purple: Spondylolisthesis



color_list = ['red' if i=='Hernia' else ('purple' if i=='Spondylolisthesis' else 'green' ) for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],

                                       c=color_list,# c - color

                                       figsize= [15,15],# figure size

                                       diagonal='hist',# histohram of each features

                                       alpha=0.5,# opacity

                                       s = 150, # size of marker

                                       marker = 'o',# marker type

                                       edgecolor= "black")

plt.show()
data['class'].value_counts()
# split dataset

x,y= data.iloc[:,: -1], data.iloc[:,-1]
# train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=1)
# import KNN algorithm

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
# fit and predict dataset

knn.fit(x_train,y_train)

pred = knn.predict(x_test)
# print accuracy

print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test))
# find best parameter

neighbor = np.arange(1,25)

train_accuracy = []

test_accuracy = []



for i,k in enumerate(neighbor):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train,y_train)

    train_accuracy.append(knn.score(x_train, y_train))

    test_accuracy.append(knn.score(x_test, y_test))



# Plot

plt.figure(figsize=[13,8])

plt.plot(neighbor, test_accuracy, label = 'Testing Accuracy')

plt.plot(neighbor, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neighbor)

plt.savefig('graph.png')

plt.show()



print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))