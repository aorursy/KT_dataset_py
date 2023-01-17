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

from sklearn.datasets import load_iris

iris = load_iris()
dir(iris)
iris.feature_names
df = pd.DataFrame(iris.data, columns = iris.feature_names)

df.head()
iris.target_names
df['target'] = iris.target
df[df.target==1].head()
from matplotlib import pyplot as plt
%matplotlib inline
df0 = df[df.target==0]

df1 = df[df.target==1]

df2 = df[df.target==2]
plt.xlabel('sepal length (cm)')

plt.ylabel('sepal width (cm)')

plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')

plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='red',marker='*')
plt.xlabel('petal length (cm)')

plt.ylabel('petal width (cm)')

plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='green',marker='+')

plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='red',marker='*')
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

df.head()
from sklearn.model_selection import train_test_split
X = df.drop(['target','flower_name'], axis='columns')

X.head()
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
len(X_train)
len(X_test)
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)
model.score(X_test, y_test)