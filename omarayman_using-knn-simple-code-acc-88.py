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
data = pd.read_csv('../input/column_2C_weka.csv')

data.info()
data.head()
data.describe()
from sklearn.model_selection import train_test_split
x = data.drop('class',axis=1)
y  = data['class']
xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(xTrain,yTrain)
knn.predict(xTest)
knn.score(xTest,yTest)
import matplotlib.pyplot as plt
n = np.arange(1,25)
test_accuracy = []
train_accuracy = []
for i,k in enumerate(n):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xTrain,yTrain)
    test_accuracy.append(knn.score(xTest,yTest))
    train_accuracy.append(knn.score(xTrain,yTrain))
plt.figure(figsize=[13,8])
plt.plot(n, test_accuracy, label = 'Testing Accuracy')
plt.plot(n, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('k value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(n)
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
knn = KNeighborsClassifier(n_neighbors=18)
knn.fit(xTrain,yTrain)
knn.predict(xTest)
knn.score(xTest,yTest)