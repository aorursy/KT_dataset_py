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
dataset = pandas.read_csv('../input/Iris.csv');
dataset.info
dataset.shape
dataset.describe()
dataset.groupby('Species').size()
# box and whisker plots

dataset.plot(kind='box', subplots=True, sharex=False, sharey=False)

plt.show()
# histograms

dataset.hist()

plt.show()


# scatter plot matrix

scatter_matrix(dataset)

plt.show()
array = dataset.values

X = array[:,0:4]

Y = array[:,4]

validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric

seed = 7

scoring = 'accuracy'