# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
names = ['separ-length','separ-width','petal-length','petal-width','class']
dataset = pd.read_csv("../input/iris.data.csv",names=names)
#show data struct
dataset.head(10)
#show data shape
dataset.shape
#show data describe
dataset.describe()
#show count per class
dataset.groupby('class').size()
#show box plot
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()
#show histogram
dataset.hist()
plt.show()
# show scatter matrix
pd.plotting.scatter_matrix(dataset,figsize=(10,10))
plt.show()
#split dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
seed = 1
#print(X.shape)
#print(Y.shape)

validation_size = 0.2
X_train,X_validation,Y_train_str,Y_validation_str=train_test_split(X,Y,test_size=validation_size,random_state=seed)

#change the string labels to number labels
Y_train = LabelEncoder().fit_transform(Y_train_str)
Y_validation = LabelEncoder().fit_transform(Y_validation_str)
print("X_train.shape:",X_train.shape)
print("Y_train.shape:",Y_train.shape)
print("X_validation.shape:",X_validation.shape)
print("Y_validation.shape:",Y_validation.shape)
#Multiple models
models = {}
models["LR"] = LogisticRegression()
models["LDR"] = LinearDiscriminantAnalysis()
models["RNN"] = KNeighborsClassifier()
models["CART"] = DecisionTreeRegressor()
models["NB"] =  GaussianNB()
models["SVM"] = SVC()
results = []
for key in models:
    kfold = KFold(n_splits=10, random_state=seed,shuffle=True)
    cv_results = cross_val_score(models[key], X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    print('%s: %f (%f)' %(key, cv_results.mean(), cv_results.std()))
fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()
#choice the best Algorithm:SVM
svm = SVC()
svm.fit(X_train,Y_train)
predictions = svm.predict(X_validation)
print("accuracy:\n", accuracy_score(Y_validation,predictions))
print("confusion_matrix:\n",confusion_matrix(Y_validation,predictions))
print("classification_report:\n",classification_report(Y_validation,predictions))

