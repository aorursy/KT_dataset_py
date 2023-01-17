# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
open_file = pd.read_csv("../input/banana.csv",sep=",")

print(open_file.head())

print(open_file.shape)
print(open_file.isnull().values.any())
print(open_file.describe())
import matplotlib.pyplot as plt

plt.scatter(open_file['At1'],open_file['At2'])

plt.show()
file = open_file[['At1','At2']]

print(file.head())
correlation = file.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlation, vmin=-1, vmax=1)

fig.colorbar(cax)

names=["At1","At2"]

ticks = np.arange(0,2,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names)

ax.set_yticklabels(names)

plt.show()
from sklearn.cross_validation import train_test_split

train,test = train_test_split(open_file,test_size=0.3)

features_train = train[['At1','At2']]

features_test = test[['At1','At2']]

labels_train = train.Class

labels_test = test.Class

print(labels_test.head())

print(train.shape)

print(test.shape)
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

training = clf.fit(features_train,labels_train)

predictions = clf.predict(features_test)

print(predictions)

print("Accuracy:",clf.score(features_test,labels_test))
from sklearn import svm

clf = svm.SVC(kernel='rbf')

training = clf.fit(features_train,labels_train)

predictions = clf.predict(features_test)

print(predictions)

print("Accuracy:",clf.score(features_test,labels_test))
from sklearn import svm

clf = svm.SVC(kernel='linear')

training = clf.fit(features_train,labels_train)

predictions = clf.predict(features_test)

print(predictions)

print("Accuracy:",clf.score(features_test,labels_test))
from sklearn import tree

clf = tree.DecisionTreeClassifier()

training = clf.fit(features_train,labels_train)

predictions = clf.predict(features_test)

print(predictions)

print("Accuracy:",clf.score(features_test,labels_test))
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()

training = clf.fit(features_train,labels_train)

predictions = clf.predict(features_test)

print(predictions)

print("Accuracy:",clf.score(features_test,labels_test))
from sklearn.ensemble import BaggingClassifier

clf = BaggingClassifier(n_estimators=100, random_state=7)

boosted = clf.fit(features_train,labels_train)

prediction = clf.score(features_test,labels_test)

print("Accuracy:",prediction)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=7)

boosted = clf.fit(features_train,labels_train)

prediction = clf.score(features_test,labels_test)

print("Accuracy:",prediction)
from sklearn import svm

clf = svm.SVC(kernel='rbf',C=10,gamma='auto')

training = clf.fit(features_train,labels_train)

predictions = clf.predict(features_test)

print(predictions)

print("Accuracy:",clf.score(features_test,labels_test))