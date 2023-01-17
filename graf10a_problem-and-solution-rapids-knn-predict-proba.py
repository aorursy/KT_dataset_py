%%time



import sys

!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
from cuml.neighbors import KNeighborsClassifier, NearestNeighbors

import cuml; cuml.__version__
from sklearn.datasets import make_blobs

from sklearn.model_selection import train_test_split



X, y = make_blobs(n_samples=100, 

                  centers=5,

                  cluster_std=5.0,

                  n_features=4)



knn = KNeighborsClassifier(n_neighbors=10)



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)



knn.fit(X_train, y_train)
X_test.shape, y_test.shape
y_test
knn.predict(X_test)
knn.predict_proba(X_test)
import numpy as np



KNN=10

batch=5



clf = NearestNeighbors(n_neighbors=KNN)



clf.fit(X_train)



distances, indices = clf.kneighbors(X_test)



ct = indices.shape[0]



pred = np.zeros((ct, KNN),dtype=np.int8)



probabilities = np.zeros((ct, len(np.unique(y_train))),dtype=np.float32)



it = ct//batch + int(ct%batch!=0)



for k in range(it):

    

    a = batch*k; b = batch*(k+1); b = min(ct,b)

    pred[a:b,:] = y_train[ indices[a:b].astype(int) ]

    

    for j in np.unique(y_train):

        probabilities[a:b,j] = np.sum(pred[a:b,]==j,axis=1)/KNN
probabilities