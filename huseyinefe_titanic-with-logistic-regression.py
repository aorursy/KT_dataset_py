# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/titanic/train.csv")

train.info()
train1=train.drop(["PassengerId","Name","Sex","Ticket","Cabin","Embarked"],axis=1)
train1.dropna(axis=0,inplace=True)



X_train=train1.drop(["Survived"],axis=1)

Y_train=train1.Survived.values.reshape(-1,1)
sns.countplot(Y_train.reshape(714, ),palette="icefire")

plt.show()
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(X_train,Y_train,test_size=0.10,random_state=42)

print("x train shape:{} ".format(x_train.shape))

print("y_train shape:{} ".format(y_train.shape))

print("x_val shape:{} ".format(x_val.shape))

print("y_val shaoe:{} ".format(y_val.shape))
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

y_predict=lr.predict(x_val)

cost_list=[]

cost_list.append(y_predict)
print("train score:{}".format(lr.score(x_train,y_train)))

print("test score:{}".format(lr.score(x_val,y_val)))
from sklearn.metrics import confusion_matrix



confusion_mtx=confusion_matrix(y_val,y_predict)

f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(confusion_mtx,annot=True,linewidths=0.01,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)

plt.xlabel("Predicted")

plt.ylabel("Real")

plt.title("Corroleation")

plt.show()
test=pd.read_csv("../input/titanic/test.csv")

gs=pd.read_csv("../input/titanic/gender_submission.csv")
test.drop(["PassengerId"],axis=1,inplace=True)
test_data=pd.concat([gs,test],axis=1)
test_data
test_data.drop(["PassengerId","Name","Sex","Ticket","Cabin","Embarked"],axis=1,inplace=True)
test_data.dropna(axis=0,inplace=True)
test_data.info()
xtest=test_data.drop(["Survived"],axis=1)

ytest=test_data.Survived.values.reshape(-1,1)
sns.countplot(ytest.reshape(331, ),palette="icefire")

plt.show()
from sklearn.model_selection import train_test_split

x_train1,x_test1,y_train1,y_test1 = train_test_split(xtest,ytest,test_size=0.10,random_state=42)

print("x train1 shape: ",x_train1.shape)

print("x test1 shape: ",x_test1.shape)

print("y train1 shape: ",y_train1.shape)

print("y test1 shape: ",y_test1.shape)
lr1=LogisticRegression()

lr1.fit(x_train1,y_train1)

y_predict1=lr1.predict(x_test1)

print("train score:{}".format(lr1.score(x_train1,y_train1)))

print("test score:{}".format(lr1.score(x_test1,y_test1)))
confusion_mtx1=confusion_matrix(y_test1,y_predict1)

f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(confusion_mtx1,annot=True,linewidths=0.1,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
