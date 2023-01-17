# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



filename = "creditcard.csv"

path = "../input/"



data = pd.read_csv(path+filename)

print("file read successfuly")



X = data[["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]]

y = data["Class"]



X_train, X_test, y_train, y_test = train_test_split(X,y)

print("train and test sets created")



knn = KNeighborsClassifier(n_neighbors = 5,n_jobs=16)

knn.fit(X_train,y_train)

print("classifier created")

score = knn.score(X_test,y_test)

print("model evaluated")

print(score)



# Any results you write to the current directory are saved as output.