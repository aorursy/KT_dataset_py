# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/voice.csv')
data.head()
data.info()
seaborn.pairplot(data[['meanfreq', 'Q25', 'Q75', 'skew', 'centroid', 'label']],hue='label', size=3)
data = data.sample(frac=1)

data.head()
data['label'] = data['label'].map({'male':1,'female':0})
X = data.loc[:, data.columns != 'label']
y = data.loc[:,'label']
X = (X - np.min(X))/(np.max(X)-np.min(X)).values

X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)
print("test accuracy: {} ".format(logreg.fit(X_train, y_train).score(X_test, y_test)))
print("train accuracy: {} ".format(logreg.fit(X_train, y_train).score(X_train, y_train)))
from sklearn.ensemble import RandomForestClassifier

ran_for = RandomForestClassifier(n_estimators=250, max_depth=15, random_state=42)
print("test accuracy: {} ".format(ran_for.fit(X_train, y_train).score(X_test, y_test)))
print("train accuracy: {} ".format(ran_for.fit(X_train, y_train).score(X_train, y_train)))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
print("test accuracy: {} ".format(knn.fit(X_train, y_train).score(X_test, y_test)))
print("train accuracy: {} ".format(knn.fit(X_train, y_train).score(X_train, y_train)))
score = []

for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X_train, y_train)
    score.append(knn2.score(X_test, y_test))
    
plt.plot(range(1,20), score)
plt.xlabel('k values')
plt.ylabel('sccuracy')
plt.show()
from sklearn.svm import SVC

svm = SVC(random_state=42)
print("test accuracy: {} ".format(svm.fit(X_train, y_train).score(X_test, y_test)))
print("train accuracy: {} ".format(svm.fit(X_train, y_train).score(X_train, y_train)))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
print("test accuracy: {} ".format(nb.fit(X_train, y_train).score(X_test, y_test)))
print("train accuracy: {} ".format(nb.fit(X_train, y_train).score(X_train, y_train)))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
print("test accuracy: {} ".format(dt.fit(X_train, y_train).score(X_test, y_test)))
print("train accuracy: {} ".format(dt.fit(X_train, y_train).score(X_train, y_train)))
