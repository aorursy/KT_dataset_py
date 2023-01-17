# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

dataset=pd.read_csv("../input/train.csv")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



X=dataset.iloc[:,1:785].values

y=dataset.iloc[:,0:1].values









from sklearn.cross_validation import train_test_split

X_train1,X_valid1,y_train,y_valid=train_test_split(X,y,test_size=0.2)







# Any results you write to the current directory are saved as output.
from sklearn.decomposition import PCA

pca=PCA(n_components=45,svd_solver="arpack")

X_pca=pca.fit_transform(X)

from sklearn.cross_validation import train_test_split

X_train,X_valid,y_train,y_valid=train_test_split(X_pca,y,test_size=0.2)



print("---pca done---")

print(pca.components_)
print(X_pca)

print(pca.score(X))

print(pca.get_covariance())
from sklearn.neighbors import KNeighborsClassifier

print (X_train.shape)
classifier=KNeighborsClassifier(n_neighbors=4,weights="distance",algorithm="kd_tree",n_jobs=2,leaf_size=10)

print("---classifier initalized----")
classifier.fit(X_train,y_train)

pred=classifier.predict(X_valid)

print("------Training Done------")

from sklearn.metrics import precision_score

precise_score=precision_score(y_valid,pred,average="micro")

print(precise_score) 