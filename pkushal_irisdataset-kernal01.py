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
import pandas

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
dataset = pandas.read_csv('../input/Iris.csv')
# shape

print(dataset.shape)

print ("\n")



# head

print(dataset.head(20))

print ("\n")



# descriptions

print(dataset.describe())

print ("\n")



# class distribution

print(dataset.groupby('Species').size())

print ("\n")



#Data Visualization:

dataset.iloc[0:, 1:5].plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

visual = dataset.iloc[0:, 1:4]

#visual.plot(kind='box', subplots=True, layout=(2,2))



dataset.iloc[0:, 1:5].hist()

plt.show()
# scatter plot matrix

scatter_matrix(dataset.iloc[0:, 1:5])

plt.show()

# Evaluate some models:



# Split-out validation dataset

array = dataset.values

X = array[:,1:5]

Y = array[:,5]



validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)



# Test Harness

# Test options and evaluation metric

seed = 7

scoringMethod = 'accuracy'



# Spot Check Algorithms

models = []

models.append(('LogisticRegression - LR', LogisticRegression()))

models.append(('LinearDiscriminantAnalysis - LDA', LinearDiscriminantAnalysis()))

models.append(('KNeighborsClassifier - KNN', KNeighborsClassifier()))

models.append(('DecisionTreeClassifier - CART', DecisionTreeClassifier()))

models.append(('GaussianNB - NB', GaussianNB()))

models.append(('SVC -SVM', SVC()))

# evaluate each model in turn

results = []

names = []

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    

    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoringMethod)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

    

# Compare Algorithms

print ("\n")

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
# here we have found a best algorithm and testing that model against the validation inputs



# Make Prediction

# Make predictions on validation dataset

knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

predictions = knn.predict(X_validation)

print(accuracy_score(Y_validation, predictions))

print ("\n")

print(confusion_matrix(Y_validation, predictions))

print ("\n")

print(classification_report(Y_validation, predictions))





print ("testing SVM algorithm for fun")



# chose

svm = SVC()

# trained the data here

svm.fit(X_train, Y_train)

# making the predictions here

predictions = svm.predict(X_validation)

# checking the prediction result here

print(accuracy_score(Y_validation, predictions))

print ("\n")

print(confusion_matrix(Y_validation, predictions))

print ("\n")

print(classification_report(Y_validation, predictions))