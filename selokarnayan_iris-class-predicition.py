# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pandas import read_csv

from pandas.plotting import scatter_matrix

from matplotlib import pyplot

# Load dataset

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

dataset = read_csv(url, names=names)

# box and whisker plots

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

pyplot.show()

print(dataset.describe())

# histograms

dataset.hist()

pyplot.show()

# scatter plot matrix

scatter_matrix(dataset)

pyplot.show()
dataset.describe()

dataset.head(10)
dataset.groupby('class').size()
from sklearn.model_selection import train_test_split

array = dataset.values

X = array[:,0:4]

y = array[:,4]

X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.30, random_state=1)
#First Model - Linear Regression

# compare algorithms

from pandas import read_csv

from matplotlib import pyplot

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

# Load dataset

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

dataset = read_csv(url, names=names)

# Split-out validation dataset

array = dataset.values

X = array[:,0:4]

y = array[:,4]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

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

	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

	results.append(cv_results)

	names.append(name)

	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

        

# Compare Algorithms

pyplot.boxplot(results, labels=names)

pyplot.title('Algorithm Comparison')

pyplot.show()
# making predictions

from pandas import read_csv

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from pandas.plotting import scatter_matrix

from matplotlib import pyplot

# Load dataset

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

dataset = read_csv(url, names=names)

# Split-out validation dataset

array = dataset.values

X = array[:,0:4]

y = array[:,4]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Make predictions on validation dataset

model = SVC(gamma='auto')

model.fit(X_train, Y_train)

predictions = model.predict(X_validation)

# Evaluate predictions

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))

print(predictions)
#Above is the prediction for IRIS validation dataset which we splitted earlier