import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp

import matplotlib as mp

import sklearn as sk

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load libraries

from pandas import read_csv

from pandas.plotting import scatter_matrix

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

dataset = pd.read_csv('/kaggle/input/iris-dataset/iris.data.csv', names=names) # load data 

print(dataset.shape) # (rows, column)

print(dataset.head(10)) # first 10 rows

print(dataset.describe()) # statistic description of the data

print(dataset.groupby('class').size()) # class distribution
# data visualization, understanding the behavior of each attribute using univariate plots, 



fig = plt.figure()

# Create an axes instance

ax = fig.add_axes([0,0,2,2])

# Create the boxplot

labels = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']

bp = ax.boxplot([dataset['sepal-length'], dataset['sepal-width'], dataset['petal-length'], dataset['petal-width']],

                     notch=True,  # notch shape

                     vert=True,  # vertical box alignment

                     patch_artist=True,  # fill with color

                     labels=labels,  # will be used to label x-ticks)

                     showmeans=True, meanline=True, meanprops = dict(linestyle='--', linewidth=2.5, color='green')) # show the mean with green line

ax.set_title('box and whisker plots')



# fill with colors

colors = ['pink', 'lightblue', 'gold', 'silver']    

for box, i in zip(bp['boxes'],range(4)):

    # change outline color

    box.set(color='black', linewidth=1)

    # change fill color

    box.set(facecolor = colors[i] )

    # change hatch

    box.set(hatch = '/')

    

# adding horizontal grid lines

ax.yaxis.grid(True)

ax.set_xlabel('Iris attribute')

ax.set_ylabel('Observed values')



# histograms

dataset.hist(bins=50,figsize=(18,12), grid=True)

plt.show()
#the relation between attributes using multivariate plots

# scatter plot matrix

scatter_matrix(dataset, diagonal='kde',figsize=(18,12), grid=True, color ='green')

plt.show()



# now you can see correlations between some pairs of attributes.
# Split-out dataset into train set and test set

# Afterwards, the train set will be splited using the k-fold cross-validation technique, then we evaluate each model to get the most accurate one for this problem 

# The test set will be used to make predictions; 

array = dataset.values

X = array[:,0:4]

y = array[:,4]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True) 



# Spot Check Algorithms

models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn

results = [] # will be used afterwards to create a plot of the model evaluation results and compare the spread and the mean accuracy of each model 

names = [] 

for name, model in models:

    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True) # k=10 -> 9 training set and 1 cross-validation set, and repeat for all combinations of train-test splits.

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    

# Compare Algorithms

fig2 = plt.figure()

ax2 = fig2.add_axes([0,0,2,2])

ax2.boxplot(results, labels=names, showmeans=True, meanline=True, meanprops = dict(linestyle='--', linewidth=2.5, color='green'))

ax2.yaxis.grid(True)

ax2.set_title('Algorithm Comparison')

plt.show()



# You can see that the SVC algorithm has high accuracy of 99%, it will be used to make predictions on the test set

# This will give us an independent final check on the accuracy of the best model.
# Make predictions on test set

model = SVC(gamma='auto')

model.fit(X_train, Y_train)

predictions = model.predict(X_test)



# Evaluate predictions

print(accuracy_score(Y_test, predictions))

print(confusion_matrix(Y_test, predictions)) # The confusion matrix provides an indication of the three errors made.

print(classification_report(Y_test, predictions)) # he classification report provides a breakdown of each class by precision, recall, f1-score and support