# -*- coding: utf-8 -*-

"""

Created on Wed Jan 31 14:26:25 2018



@author: Santosh.Bothe

"""

# Load libraries

import pandas

from pandas.tools.plotting import scatter_matrix

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



# Load dataset

#Code for ACD Log D 



url = "../input/Concrete_Data.csv"

names = ['Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water', 'Superplasticizer', 'CoarseAggregate', 'Fine Aggregate', 'Age', 'CompressiveStrength']

dataset = pandas.read_csv(url, names=names)



# descriptions

print(dataset.describe())

# class distribution

dataset.groupby('CompressiveStrength').size()

plt.figure;

fig = plt.gcf()

fig.set_size_inches(18.5, 10.5)

bp=dataset.boxplot()

 #(kind='box', subplots=True, layout=(4,2), sharex=False, sharey=False)

plt.show()

# Split-out validation dataset

array = dataset.values

X = array[:,0:8]

Y = array[:,8]

validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)



# Test options and evaluation metric

#seed = 20

scoring = 'accuracy'

# Spot Check Algorithms

print("Results with ACD Log D")

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

# evaluate each model in turn

results = []

names = []

print('', '')

for name, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=seed)

	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

 #   msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	msg = "Name of Algorithm : %s: Accuracy:  %f (Stendard Deviation %f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)



# Code for ACD Log P

    # Load dataset

#

#url = "d:\model\WithLogP.csv"

#names = ['Dose', 'Molecular Weight', 'Bioavailability', 'IC50 at target', 'IC50 CYP corrected', 'IC50 hERG', 'Cmax (uM/L)', 'Heavy atom number', 'ACD logP', 'BSEP transporter (Taurocholate) IC50 ', 'PPB fraction', 'Result']

#dataset = pandas.read_csv(url, names=names)

## class distribution

#dataset.groupby('Result').size()

## Split-out validation dataset

#array = dataset.values

#X = array[:,0:11]

#Y = array[:,11]

#validation_size = 0.20

#seed = 7

#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#

## Test options and evaluation metric

##seed = 20

#scoring = 'accuracy'

## Spot Check Algorithms

#models = []

#models.append(('LR', LogisticRegression()))

#models.append(('LDA', LinearDiscriminantAnalysis()))

#models.append(('KNN', KNeighborsClassifier()))

#models.append(('CART', DecisionTreeClassifier()))

#models.append(('NB', GaussianNB()))

#models.append(('SVM', SVC()))

## evaluate each model in turn

#results = []

#names = []

#print("\nResults with ACD Log P")   

#

#for name, model in models:

#	kfold = model_selection.KFold(n_splits=10, random_state=seed)

#	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

#	results.append(cv_results)

#	names.append(name)

# #   msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

#	msg = "Name of Algorithm : %s: Accuracy:  %f (Stendard Deviation %f)" % (name, cv_results.mean(), cv_results.std())

#	print(msg)

## Make predictions on validation dataset

#LR= LogisticRegression()

#LR.fit(X_train, Y_train)

#predictions = LR.predict(X_validation)



#print("\nAccuracy Score of LogisticRegression Y Validation : ", accuracy_score(Y_validation, predictions))

#print("\n Cconfusion_matrix Y: \n", confusion_matrix(Y_validation, predictions))



#print(classification_report(Y_validation, predictions))