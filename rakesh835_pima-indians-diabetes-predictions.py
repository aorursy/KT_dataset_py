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
df=pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

df.head(10)
df.describe()
df.shape
target=df.Outcome
df.drop("Outcome", inplace=True, axis=1)
df.shape[1]
for i in range(df.shape[1]):

    df.iloc[i].plot(kind="bar")
df.isnull().sum()
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(df, target, test_size=0.2)
scaler= MinMaxScaler()

x_train=scaler.fit_transform(x_train)

x_test=scaler.transform(x_test)
from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV 

from sklearn.metrics import accuracy_score
#clf=SVC(gamma="scale")

#clf.fit(x_train, y_train)
Cs = [0.001, 0.01, 0.1, 1, 10, 100]

gammas = [0.001, 0.01, 0.1, 1, 10]

param_grid = {'C': Cs, 'gamma' : gammas}

grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3)

grid_search.fit(x_train, y_train)
pred=clf.predict(x_test)

print(accuracy_score(pred, y_test))