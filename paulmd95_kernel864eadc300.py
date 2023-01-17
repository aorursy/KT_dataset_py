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
import seaborn

import matplotlib.pyplot as plt
test=pd.read_csv("../input/test.csv",header=None)

test.drop([0],inplace=True)

print(test.head(5))
train_unprocessed=pd.read_csv("../input/train.csv",header=None)

train_unprocessed.drop([0],inplace=True)

print(train_unprocessed.head(5))
target=train_unprocessed[10]

print(target.head(5))
seaborn.pairplot(train_unprocessed,vars=[1,2,3,5,7],hue=10)

plt.show()
plt.hist(train_unprocessed[3])

plt.show()
train_selected=pd.DataFrame()

train_selected=train_unprocessed[[1,2,3,5,7]]

print(train_selected.head(5))

test=test[[1,2,3,5,7]]
from sklearn.linear_model import LogisticRegressionCV

lr=LogisticRegressionCV()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train_selected,target,test_size=0.2)
model=lr.fit(X_train,y_train)
p=lr.predict(X_test)
print(p)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,p))

print(classification_report(y_test,p))