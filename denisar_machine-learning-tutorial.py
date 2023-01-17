# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#read csv file

data = pd.read_csv('../input/Iris.csv')

data.head(10)
#check dimensions of the dataset

data.shape
data.info()
#statistical summary of the data

data.describe()
data.groupby('Species').size()
import matplotlib.pyplot as plt

#box and whisker plots

data.plot(kind='box', subplots=True, layout=(1,5), sharex=False, figsize=(15,4))

plt.show()
#delete Id column

df = data.drop(["Id"], axis=1)
df.shape
#histogram

df.plot.hist(alpha=0.8, figsize=(8,5))
from pandas.plotting import scatter_matrix

#scatter plot matrix

scatter_matrix(df, alpha=0.8, figsize=(14,10))

plt.show()
from sklearn import model_selection

#create validation set

array = df.values

X = array[:,0:4]

Y = array[:,4]

validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#10-fold cross validation

seed = 7

scoring = 'accuracy'
#build the models

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

#check which of the models will have the highest accuracy

models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr' )))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN',KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB',GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

#evaluate each model

results = []

names = []

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())

    print(msg)
#compare algorithms

fig = plt.figure()

fig.suptitle('Algorithms Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Make predictions on validation data

#using svm since it gave the highest performance

svm = SVC()

svm.fit(X_train, Y_train)

predictions = svm.predict(X_validation)

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))