# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/column_2C_weka.csv')

data.sample(5)
data.info()
# Investigate the correlation between the features:

data.corr()
# Visulization of the features:

color_list = ['orange' if i=='Abnormal' else 'blue' for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'], c=color_list, figsize= [12,12], diagonal='hist', alpha=0.3, s = 200, marker = '.')

plt.show()
# Change classes to Abnormal = 1, Normal = 0:

data['class'] = [1 if each == 'Abnormal' else 0 for each in data['class']]

data.sample(5)
y = data['class'].values

x_data = data.drop(['class'], axis=1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

x.head()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)
print('x_train shape: ', x_train.shape)

print('y_train shape: ', y_train.shape)

print('x_test shape: ', x_test.shape)

print('y_test shape: ', y_test.shape)
from sklearn.neighbors import KNeighborsClassifier



# Creating model with the k value of 3

knn = KNeighborsClassifier(n_neighbors = 3)



# Training the model

knn.fit(x_train, y_train)
# Predicting our y values using our KNN model and x_test:

prediction = knn.predict(x_test)



# Comparing y_prediction and y_test values:

datashow = {'y_prediction': prediction, 'y_test': y_test}

d_new = pd.DataFrame(datashow)

d_new.T   # For the ease of reading I implemented transpose of our dataset.
print('Score of the model for k=3: ', knn.score(x_test, y_test))
score_list=[]

for each in range(1,25):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2 = knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))



plt.figure(figsize=(10,5))

plt.plot(range(1,25), score_list)

plt.xlabel('k values range')

plt.ylabel('Score values')

plt.show()
# Retraining and predicting our dataset with the best k value (k=19):

knn3 = KNeighborsClassifier(n_neighbors=19)

knn3.fit(x_train,y_train)

y_prediction = knn3.predict(x_test)

print('Score of the model for k=19: ', knn3.score(x_test, y_test))
d = {'y_prediction': y_prediction, 'y_test': y_test}

data01 = pd.DataFrame(data=d)

data01.T
correct = 0

false = 0

for each in range(1,len(data01)):

    if data01.y_test[each] == data01.y_prediction[each]:

        correct = correct + 1

    else:

        false = false + 1



print('correct predictions = ', correct)

print('false predictions = ', false)