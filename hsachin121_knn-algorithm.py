# import libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import numpy as np

import os

print(os.listdir("../input"))

data = pd.read_csv('../input/column_2C_weka.csv')
# view data

data.head()
data['class'].unique()
data.info()
# you split class as abnormal and normal

A = data[data['class'] == 'Abnormal']

N = data[data['class'] == 'Normal']
N.info()
#visualization

plt.scatter(A.pelvic_incidence, A.pelvic_radius, color = 'purple', label = 'Abnormal',alpha = 0.5)

plt.scatter(N.pelvic_incidence, N.pelvic_radius, color = 'orange', label = 'Normal', alpha = 0.5)

plt.xlabel('pelvic_incidence')

plt.ylabel('pelvic_radius')

plt.legend()

plt.show()
# abnormal and normal are string. So you transform integer or float.

data['class'] = [0 if each == 'Abnormal' else 1 for each in data['class']]
# determine feature and feature class.

y = data['class'].values

x_data = data.drop(['class'], axis=1)
# normalization

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

# (x-min(x))/(max(x)-min(x))
#train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 1)
#knn model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 4)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
print('{} nın score: {}'.format(3,knn.score(x_test,y_test)))
# find the most appropriate k value

score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train, y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list,color='purple')

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()