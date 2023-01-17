# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier # Kneighbors Classifier

from sklearn.model_selection import train_test_split # Train and Test dataset split

from sklearn.cluster import KMeans # KMeans cluster for the clustering algorithm

from sklearn.metrics import accuracy_score #Accuracy Score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# Read the training data file

df = pd.read_csv('../input/train.csv', low_memory = False, header=None, skiprows=1)



# separating features from the labels

X = df.drop(0, axis=1)

y = df[0]



#splitting dataset in train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# merging the training dataset for clustering algorithm

raw_data = pd.concat([y_train, X_train], axis=1)



#reducing original dataset into a set of 100 concentrated(centers obtained through Kmeans) records per digit.

# The concentrated dataset will have 1000 rows, hoping it will be better at predicting the digits with KNeighbors later.

cldf = pd.DataFrame();

for i in range(0,10):

    raw_data_X = raw_data[raw_data[0]==i]

    digit = raw_data_X.drop(0, axis=1)

    raw_data_y = raw_data_X[0]

    cluster = KMeans(n_clusters = 100, random_state=0, max_iter = 20).fit(digit)

    tempdf = pd.DataFrame(data=cluster.cluster_centers_)

    tempdf[784] = raw_data_y.iloc[0]

    cldf = cldf.append(tempdf)



# splitting the labels and features from the clustered dataset to help train the Kneighbors

cldf_X = cldf.drop(784, axis=1)

cldf_y = cldf[784]



# training 1nearest neighbor classifier as part of the nearest centroidal approach

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(cldf_X, cldf_y)



# predictions

pred = knn.predict(X_test)

#accuracy

accuracy_score(y_test, pred)