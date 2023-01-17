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

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.

import warnings

# filter warnings

warnings.filterwarnings('ignore')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv("../input/heart-disease/heart.csv")

data.head()

data_x = data.drop(["target"],axis = 1)

data_x.head()
data_y = data.loc[:,["target"]]

data_y.head()

# if the target  Y / N instead of 1 and 0 we should use data.diagnosis = [1 if each == "Y" else 0 for each in data.diagnosis]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y , test_size=0.15, random_state=42)

from sklearn import linear_model

logreg= linear_model.LogisticRegression(random_state = 42,max_iter= 50)

print("test accuracy: {} ".format(logreg.fit(X_train, Y_train).score(X_test, Y_test)))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 60)

knn.fit(X_train,Y_train)

prediction = knn.predict(X_test)

print("{} nn score  : {} ".format(60,knn.score(X_test,Y_test)))

score_list = [] 

for each in range(1,150):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(X_train,Y_train)

    score_list.append(knn2.score(X_test,Y_test))



  

plt.plot(range(1,150),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
from sklearn.svm import SVC

svm = SVC(random_state = 1)

print("test accuracy: {} ".format(svm.fit(X_train, Y_train).score(X_test, Y_test)))
from sklearn.naive_bayes import GaussianNB

naive = GaussianNB()

naive.fit(X_train,Y_train)

print("test accuracy: {} ".format(naive.fit(X_train, Y_train).score(X_test, Y_test)))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(X_train,Y_train)

print("test accuracy: {} ".format(dt.score(X_test, Y_test)))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000 , random_state = 1)

rf.fit(X_train,Y_train)

print("test accuracy: {} ".format(rf.score(X_test, Y_test)))
y_true = Y_test

y_head = rf.predict(X_test)





from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_head)

print(cm)
import seaborn as sns

plt.figure(figsize = (7,7))

sns.heatmap(cm,annot = True)

plt.xlabel("y_head")

plt.ylabel("y_true")