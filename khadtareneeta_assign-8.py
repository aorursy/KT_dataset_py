!pip install -U scikit-learn
from sklearn import model_selection
import sys

print('python:{}'.format(sys.version))
import scipy

import numpy

import matplotlib

import pandas as pd

import sklearn
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
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

dataset=pandas.read_csv(url, names=names)
print(dataset.hea)