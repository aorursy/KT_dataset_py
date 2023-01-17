# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/social-network-ads/Social_Network_Ads.csv")
data.head()
data.isnull().sum()
from sklearn import preprocessing
data.head()
from sklearn.model_selection import train_test_split

data=pd.get_dummies(data,columns=["Gender"])   # one hot encoding
x=data.drop(["User ID","Purchased"],axis=1)

y=data["Purchased"]
x.head()
y.head()
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30,random_state=0)     # 0.30 or 30% is the size of test
xtrain.head()
xtrain["Age"]=pd.cut(xtrain["Age"],bins=3,labels=False)
xtrain.head()
xtrain["EstimatedSalary"]=pd.cut(xtrain["EstimatedSalary"],bins=10,labels=False)
xtrain.head()
xtrain.head()
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(xtrain,ytrain)
xtest["Age"]=pd.cut(xtest["Age"],bins=3,labels=False)
xtest["EstimatedSalary"]=pd.cut(xtest["EstimatedSalary"],bins=10,labels=False)
xtest.reset_index(inplace=True)
xtrain.head()
xtest.head()
ypred=lg.predict(xtest.drop("index",axis=1))
ypred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_matrix(ytest,ypred)
accuracy_score(ytest,ypred)
# Accuracy obtained is 83 % 