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
import numpy as np

import pandas as pd

import seaborn as sns

sns.set_palette('husl')

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')


from sklearn.datasets import load_iris

iris=load_iris()

print(iris.feature_names)

print(iris.target_names)
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
print(iris.data.shape)

iris.target.shape

x=iris.data

y=iris.target

from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.4,random_state=1)
knn=KNeighborsClassifier(n_neighbors=2)

logreg=LogisticRegression()

logreg.fit(xTrain,yTrain)

yPred =logreg.predict(xTest)





print(metrics.accuracy_score(yPred,yTest))
k_range=range(1,26)

a=[]

for k in k_range:

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(xTrain,yTrain)

    y_pred=knn.predict(xTest)

    a.append(metrics.accuracy_score(y_pred,yTest))
import matplotlib.pyplot as plt

plt.plot(k_range,a)
knn=KNeighborsClassifier(n_neighbors=10)

knn.fit(xTrain,yTrain)

y_pred=knn.predict(xTest)

metrics.accuracy_score(y_pred,yTest)
### 