# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #math plots 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#import the Iris datasets with pandas

mydataset = pd.read_csv('../input/Iris.csv')
X = mydataset.iloc[:,[1,2,3,4]].values



#X = mydata[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

X
y = mydataset['Species']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=0)



print('There are {} samples in the Training Set and {} samples in the Test Set'.format(X_train.shape[0], X_test.shape[0]))
from sklearn.svm import SVC



model = SVC(kernel = 'rbf', random_state=0, gamma=.10, C=1.0)

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

y_predicted_train = model.predict(X_train)



y_predicted
model.score(X_train, y_train)
model.score(X_test, y_test)
from sklearn.cluster import KMeans



X = mydataset.iloc[:,[1,2,3,4]].values

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(X)
y_kmeans
#visualising the clusters

plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0, 1], s = 100, c='red', label = 'Iris_setosa')

plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1, 1], s = 100, c='blue', label = 'Iris_versicolour')

plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2, 1], s = 100, c='green', label = 'Iris_virginica')



#plotting the centroids of the clusters

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='^', s = 150, c = 'black', label = 'Centroids')



plt.legend()