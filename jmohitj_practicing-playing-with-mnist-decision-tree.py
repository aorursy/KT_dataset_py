# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

print(df_train.shape)
print(df_test.shape)
X = []
y = []
for row in df_train.iterrows() :
    label = row[1][0] # label (the number visible in the image)
    image = list(row[1][1:]) # image information as list, without label
    image = np.array(image) / 255
    X.append(image)
    y.append(label)

X = np.array(X)
y = np.array(y)
print(len(X))
print(len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(len(X_train), len(y_train))
print(X_train[1].shape)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred[0:20], ".....")
print(y_test[0:20], ".....")
print(metrics.accuracy_score(y_test, y_pred))
X_new = []
for row in df_test.iterrows() :
    image = list(row[1])
    image = np.array(image) / 255
    X_new.append(image)
X_new = np.array(X_new)
print(len(X_new))
print(len(df_test))
y_new_pred = clf.predict(X_new)
print(y_new_pred)
df_sub = pd.DataFrame(list(range(1,len(X_new)+1)))
df_sub.columns = ["ImageID"]
df_sub["Label"] = y_new_pred
df_sub.to_csv("submission.csv", sep=",", header=True, index=False)
print(df_sub.head())