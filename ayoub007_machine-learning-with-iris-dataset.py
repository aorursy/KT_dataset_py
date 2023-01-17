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
#Iport Libraries :
import sys
import scipy 
import numpy #linear algebra
import matplotlib #visualization
import pandas #data mining and framing
import sklearn #ml libraries

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
dataset = pd.read_csv('../input/iris.csv')
#Summarize the Dataset
print(dataset.shape)
#Describe the dataset (may give you good general informations about dataset )
dataset.describe()
#Have a look at the first 15 elements :
dataset.head(15)
#class distribution
dataset.groupby('iris').size() #groupby : regroup elements from the same class (iris-setosa, iris-versicolor)
#box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
#Transform categorical data (iris column) into numerical just for some other visualisations :
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classes = labelencoder.fit_transform(dataset['iris'])
#Effect of length on iris class:
from mpl_toolkits.mplot3d import Axes3D #used for 3D visualisations !!!
# scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(dataset['sepal length'], dataset['petal length'], classes, color='#ef1234')
plt.show()
#effect of width on iris class :
from mpl_toolkits.mplot3d import Axes3D #used for 3D visualisations !!!
# scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(dataset['sepal width'], dataset['petal width'], classes, color='#ef1234')
plt.show()
#More visualization via seaborn :
%matplotlib inline
import seaborn as sns
sns.barplot(x="iris", y="petal length", data=dataset)
plt.scatter(dataset["iris"] ,dataset["sepal width"] , alpha=0.2)
plt.show()
# histograms
dataset.hist()
plt.show()
# scatter plot matrix
scatter_matrix(dataset)
plt.show()
#this can help to find correlation between features
#in some cases it is recommended to drop some features which have high correlation
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
# Classification and performance :
#using GridSearchCV to find best parameters for RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)
predictions = clf.predict(X_validation)
print(accuracy_score(y_validation, predictions))
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

seed = 7
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RdmForest' , RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    #SVM gave the best performences
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
