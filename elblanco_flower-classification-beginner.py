import pandas as pd

from pandas.plotting import scatter_matrix

import numpy as np

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
dataset = pd.read_csv('../input/Iris.csv')

dataset = dataset.drop(['Id'],axis=1)
#shape of the dataset

dataset.shape
#head of the dataset

print(dataset.head(6))
#Statistical Description

dataset.describe()
print(dataset.groupby('Species').size())
#BOX PLOT

dataset.plot(kind='box', subplots=True, layout=(2,2) ,sharex = False, sharey = False)

plt.rcParams['figure.figsize']=(10,10)

plt.show()
#Histogram

dataset.hist()

plt.rcParams['figure.figsize']=(10,10)

plt.show()
#ScatterPlot

scatter_matrix(dataset)

plt.rcParams['figure.figsize']=(10,10)

plt.show()
#Train Test Split

X = dataset.values[:,0:4]

Y = dataset.values[:,4]

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size = 0.2,random_state = 7)

seed = 7

scoring = 'accuracy'

# Spot Check Algorithms

models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn

results = []

names = []



for name, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=seed)

	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)

# Compare Algorithms

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
# Make predictions on validation dataset

knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

predictions = knn.predict(X_test)

print(accuracy_score(Y_test, predictions))

print(confusion_matrix(Y_test, predictions))

print(classification_report(Y_test, predictions))