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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn import datasets

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split,cross_val_score



iris=datasets.load_iris()



X=iris.data[:,:2]

Y=iris.target



X.shape



X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,stratify=Y,random_state=0)



print(X_train.shape,Y_train.shape)



from sklearn.neighbors import KNeighborsClassifier



knn_3_clf=KNeighborsClassifier(n_neighbors=3)

knn_5_clf=KNeighborsClassifier(n_neighbors=5)



knn_3=knn_3_clf.fit(X_train,Y_train)

knn_5=knn_5_clf.fit(X_train,Y_train)



knn_3_score=cross_val_score(knn_3_clf,X_train,Y_train,cv=10)

knn_5_score=cross_val_score(knn_5_clf,X_train,Y_train,cv=10)



print("3 Neighbor:",knn_3_score)

print("5 Neighbor:",knn_5_score)



print("Avg score 3 neighbor:",knn_3_score.mean(),"Standard Deviation:",knn_3_score.std())

print("Avg score 5 neighbor:",knn_5_score.mean(),"Standard deviation:",knn_5_score.std())



print("We now check for the ideal value of 'k' or the number of nearest neighbours for good prediction")

all_scores=[]

for i  in range(3,9,1):

  knn_clf=KNeighborsClassifier(n_neighbors=i)

  all_scores.append((i,cross_val_score(knn_clf,X_train,Y_train,cv=10).mean()))



all_scores



sorted(all_scores,key= lambda x:x[1],reverse=True)