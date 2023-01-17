# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



dataset = pd.read_csv("../input/train.csv")

dataset.shape

# Any results you write to the current directory are saved as output.

print(dataset.head())
nonnumerics = ['object']

datasetnew = dataset.select_dtypes(include = nonnumerics)

datasetnew.shape
#All the columns have numeric values

#All the cells have discrete numeric values containing the presence/absence

#Data is categorical

#Let's see the skewness in the target variable and see if there is any imbalance in data

print(dataset["label"].skew())
print(dataset["label"].value_counts())

# The target data is balancd
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

shape = dataset.shape

X_dataset = dataset.iloc[:,1:shape[1]]

y_dataset = dataset.loc[:,"label"]

print(X_dataset.shape)

print(y_dataset.shape)

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset,

                                                    test_size = 0.20,random_state =1 )



print(X_train.shape)

print(y_train.shape)

model = DecisionTreeClassifier()



model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score



accuracy = accuracy_score(y_test,predictions)

accuracy
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

shape = dataset.shape

X_train = dataset.iloc[:,1:shape[1]]

y_train = dataset.loc[:,"label"]

testdataset = pd.read_csv("../input/test.csv")



print(X_train.shape)

print(y_train.shape)

model = DecisionTreeClassifier()



model.fit(X_train, y_train)

predictions = model.predict(testdataset)
test = pd.read_csv("../input/test.csv")

#print(test["ImageId"])

print(test.index.values)

IDs = test.index.values + 1

IDs
results = pd.DataFrame({

    "ImageID":IDs,

    "Label":predictions

})

results.to_csv("decisiontree.csv", index = False)