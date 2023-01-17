# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/column_2C_weka.csv')

print(plt.style.available) # available plot styles

plt.style.use('ggplot')
# to see features and target variable

data.head()
data.info()
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
sns.countplot(x="class", data=data)

data.loc[:,'class'].value_counts()
# KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier (n_neighbors = 3)

x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

knn.fit(x,y)

prediction = knn.predict(x)

print('Prediction: {}'.format(prediction))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)

knn = KNeighborsClassifier(n_neighbors = 3)

x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

# print ('Prediction: {}'.format(prediction))

print('KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) #accuracy
neig = np.arange (1,25)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for i,k in enumerate(neig):

    #k from 1 to 25(exclude)

    knn= KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(x_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(x_train, y_train))

    #test accuracy

    test_accuracy.append(knn.score(x_test, y_test))

    

#Plot

plt.figure(figsize=[15,10])

plt.plot(neig, test_accuracy,label = 'Test Accuracy')

plt.plot(neig, train_accuracy, label = 'Train Accuracy')

plt.legend()

plt.title('-value VS accuracy-')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.savefig('GraphPlot.png')

plt.show()

print("Best accuracy is {} K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

    