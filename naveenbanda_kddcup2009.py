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
import pandas as pd
import numpy as np
dataset=pd.read_table("../input/orange_small_train.data", "\t")
appetency=pd.read_csv("../input/appetency.csv", header=None)
upselling=pd.read_csv("../input/upselling.csv", header=None)

print ("Dataset Size: "+str(dataset.shape))
dataset.head()
appetency.head()

upselling.head()
print(appetency.shape)
print(upselling.shape)

nans=[]
for i in range(1,231):
    nans.append(sum(pd.isnull(dataset['Var'+str(i)])))
print (nans)
count=0
for i in nans:
    if i>=45000:
        count+=1
print (count)
colsToDelete=[]
for i in range(0,230):
    if nans[i]>=45000:
        colsToDelete.append(i+1)
print (colsToDelete)
for i in colsToDelete:
    del dataset['Var'+str(i)]
print (dataset.shape)
column_names = list(dataset)
print (column_names)
for i in range(191,231):
    col="Var"+str(i)
    if col in column_names:
        dataset[col].fillna(0, inplace=True)
for i in range(1, 191):
    col="Var"+str(i)
    if col in column_names:
        dataset[col].fillna(dataset[col].mean(), inplace=True)
print (dataset.isnull().sum())
for i in range(191,231):
    col="Var"+str(i)
    if col in column_names:
        dataset[col] = pd.get_dummies(dataset[col])
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(dataset, appetency, test_size=0.20, random_state=42)
appetencyModel = LinearSVC(random_state = 0)
appetencyModel.fit(X_train, y_train)
prediction=appetencyModel.predict(X_test)
print (accuracy_score(y_test, prediction))
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(dataset, upselling, test_size=0.20, random_state=42)
upsellingModel = LinearSVC(random_state = 0)
upsellingModel.fit(X_train, y_train)
prediction=upsellingModel.predict(X_test)
print (accuracy_score(y_test, prediction))









