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
cans=pd.read_csv("../input/data.csv")
cans.head()
cans.tail()
cans.describe()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

bazi=cans.iloc[:,1:10]

sns.heatmap(bazi.corr(),annot = True)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

bazi=cans.iloc[:,1:12]

sns.heatmap(bazi.corr(),annot = True)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

y=cans["diagnosis"]

x=cans.iloc[:,2:32]

yint=y.map({"M":1,"B":-1})

xnew=SelectKBest(chi2, k=14).fit_transform(x, yint)
type(xnew)
newxs=pd.DataFrame(xnew)
sns.heatmap(newxs.corr(),annot=True)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

y=cans["diagnosis"]

x=cans.iloc[:,2:32]

xnew2=SelectKBest(chi2, k=6).fit_transform(x, yint)

newxs2=pd.DataFrame(xnew2)

sns.heatmap(newxs2.corr(),annot=True)
from sklearn.feature_selection import SelectFromModel

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(max_depth=8)

dt.fit(x,y)

model = SelectFromModel(dt, prefit=True)

newx=model.transform(x)
yeniler=pd.DataFrame(newx)

sns.heatmap(yeniler.corr(),annot=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(yeniler, y, test_size=0.25, random_state=4341)

from sklearn.svm import SVC

svm=SVC()

svm.fit(X_train,y_train)

ypred=svm.predict(X_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,ypred))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,ypred))