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
data=pd.read_csv("../input/train.csv")

data.head()
x=data.drop(["price_range"],axis=1)

x.head()
y=data.price_range.values

y
#Normalization

x=(x-np.min(x))/(np.max(x)-np.min(x))

x.head()
from sklearn.model_selection import train_test_split

xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.tree import DecisionTreeClassifier

DTC=DecisionTreeClassifier()

DTC.fit(xTrain,yTrain)

print("Decision Tree Values: ",DTC.score(xTest,yTest))
from sklearn.ensemble import RandomForestClassifier

RF=RandomForestClassifier(n_estimators=100,random_state=1)

RF.fit(xTrain,yTrain)

print("Random Forest Score: ",RF.score(xTest,yTest))
