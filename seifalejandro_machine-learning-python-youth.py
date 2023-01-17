# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#impor the Iris dataset with pandas

mydataset = pd.read_csv('../input/Iris.csv')
X = mydataset.iloc[:,[1,2,3,4]].values

# X = mydataset[['SepalLengthCm','SepalWitdthCm',...]]

# By convention, X is the data

X
y = mydataset['Species']

y ## By convention, y are the labels
from sklearn.model_selection import train_test_split

## X_ .. are features

## y_ .. are labels

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.9, random_state=0)



print('There are {} samples in the Training Set and {} samples in the Test Set.'\

      .format(X_train.shape[0], X_test.shape[0]))
from sklearn.svm import SVC

## This is where we train the model.

## We will use the smv machine learning method.

## SVC is a variete from the SVM method.



model= SVC(kernel='rbf', random_state=0,gamma=.10, C=1.0)

model.fit(X_train,y_train)

### after fitting, you want to do prediction. We'll predict first the test set

y_predicted= model.predict(X_test)

## you don't have to use the label, since you are trying to predict the label.

y_predicted= model.predict(X_train)
y_predicted,np.array(y_train)
model.score(X_train,y_train)

## This should be good (but less than 1.) since I used the training data
model.score(X_test,y_test)



## Depending on the amount of data used for training, the result. 

#The more data I use for training, the better results it should yield
from sklearn.cluster import KMeans



X=mydataset.iloc[:,[1,2,3,4]].values ## redundant, already had it from before

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(X)
y_kmeans

## This predicts cluster numbers 0,1,2 for each data point. 

## We said n_clusters = 3, that's why we have 3 types.
## Visualising the clusters

plt.figure(figsize=(10,10))



plt.scatter(X[y_kmeans ==0,0],X[y_kmeans ==0,1], s =100, c = 'red', label='Iris-setosa')

plt.scatter(X[y_kmeans ==1,0],X[y_kmeans ==1,1], s =100, c = 'blue', label='Iris-versicolor')

plt.scatter(X[y_kmeans ==2,0],X[y_kmeans ==2,1], s =100, c = 'green', label='Iris-virginica')



## Plotting the centroid of the clusters

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='^',s=150, c = 'black'\

           ,label = 'centroids')

plt.legend()

plt.show()