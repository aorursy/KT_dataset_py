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
data = pd.read_csv('../input/pulsar_stars.csv')
data.head()
y = data.target_class.values

x_data = data.drop(['target_class'], axis = 1 )
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

x.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)
from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train, y_train)

prediction = knn.predict(x_test)
print("{} score {}".format(3, knn.score(x_test, y_test)))
score_list = []

index = []

maxi = 0

for i in range(1,20):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(x_train, y_train)

    prediction = knn.predict(x_test)

    score_list.append(knn.score(x_test, y_test))

    index.append(i)    

plt.plot(index, score_list)

plt.xticks(index,rotation='vertical')

plt.xlabel("k values")

plt.ylabel("score")

plt.show()