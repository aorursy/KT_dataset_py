# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

df1.head()
x = df1.iloc[:,:-1].values   # Independent Variables

y = df1.iloc[:,-1:]          # Dependent Variables
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

print("\nK=3 for Accuracy: {}".format(knn.score(x_test,y_test)))
scoreList = []

for i in range(1,20):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    score = knn.score(x_test,y_test)

    scoreList.append(score)

    

plt.plot(range(1,20),scoreList)

plt.grid()

plt.show()
knn = KNeighborsClassifier(n_neighbors=18)

knn.fit(x_train,y_train)

print("\n K = 18 => Accuracy: {} ".format(knn.score(x_test,y_test)))