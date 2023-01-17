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
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

submission=pd.read_csv("../input/sample_submission.csv")
train.head()
test.head()
submission.head()
y_train=train["target"]

print(y_train.head())

x_train=train

del x_train["target"]
test_id=test["id"]

x_test=test

del x_test["id"]

del x_train["id"]
print(x_train.head())

print(x_test.head())
#Shapes

print(x_train.shape)

print(x_test.shape)
#Divding my train data to see whetther my data overfit

from sklearn import model_selection

x_train_train,x_train_test,y_train_train,y_train_test=model_selection.train_test_split(x_train,y_train)
#linear Regressor

from sklearn.linear_model import LinearRegression
clf=LinearRegression()

clf.fit(x_train_train,y_train_train)
print(clf.score(x_train_test,y_train_test))

print("Linear Model will fail badly")
#Applying Logistic Regression

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(solver='saga',max_iter=1000,C=0.05)

clf.fit(x_train_train,y_train_train)
print(clf.score(x_train_test,y_train_test))

print("Logistic Regression is giving a accuracy of 73% with test data created fom testing data")

y_predict=clf.predict(x_test)
predictions=pd.DataFrame({"id":test_id,"target":y_predict})
predictions.to_csv("Dont_overfit.csv")
#Random_Forest

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=100,n_jobs=8,max_depth=50,min_samples_split=2,random_state=1)

clf.fit(x_train_train,y_train_train)
clf.score(x_train_test,y_train_test)

from sklearn.svm import SVC

clf=SVC(random_state=1,degree=3,gamma='scale')

clf.fit(x_train_train,y_train_train)
clf.score(x_train_test,y_train_test)