# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.metrics import accuracy_score

from sklearn import model_selection

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler # for scaling data

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#loading dataset into pandas Dataframe

dataset = pd.read_csv("../input/Iris.csv")

dataset.drop("Id", axis = 1, inplace = True)

# shape

print(dataset.shape)
# head

print(dataset.head(20))
# descriptions

print(dataset.describe())
#divide the training dataset into training and testing datasets

arrays = dataset.values

X = arrays[:, 0:4]

Y = arrays[:, 4]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = 0.2, random_state = 7)



#Scaling the feautres to a similar scale

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



print(X_train)

print(X_test)
scoring = 'accuracy'

models = []

models.append(("LR", LogisticRegression()))

models.append(("LDA", LinearDiscriminantAnalysis()))

models.append(("KNN", KNeighborsClassifier()))

models.append(("CART", DecisionTreeClassifier()))

models.append(("NB", GaussianNB()))

models.append(("SVM", SVC()))



#evaluate the performance of each model in turn to determine the best model for our dataset



results = []

names = []

for name, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=7)

	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)
lda = LogisticRegression()

lda.fit(X_train, Y_train)

predictions = lda.predict(X_test)

print(accuracy_score(Y_test,predictions))
svm = SVC()

svm.fit(X_train, Y_train)

predictions = svm.predict(X_test)

print(accuracy_score(Y_test, predictions))
nb = GaussianNB()

nb.fit(X_train, Y_train)

predictions = nb.predict(X_test)

print(accuracy_score(Y_test, predictions))