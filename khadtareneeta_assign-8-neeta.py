import sys

print('Python: {}'.format(sys.version))
# scipy

import scipy

print('scipy: {}'.format(scipy.__version__))
# numpy

import numpy

print('numpy: {}'.format(numpy.__version__))
# matplotlib

import matplotlib

print('matplotlib: {}'.format(matplotlib.__version__))
# pandas

import pandas 

print('pandas: {}'.format(pandas.__version__)) 
# scikit-learn

import sklearn

print('sklearn: {}'.format(sklearn.__version__))
# scikit-learn

import sklearn

print('sklearn: {}'.format(sklearn.__version__))
from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# load the data directly from the UCI Machine Learning repository

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00194/sensor_readings_2.data"

names = ['SD_front','SD_left','class']

dataset = pandas.read_csv(url, names=names)
# load the data directly from the UCI Machine Learning repository

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

dataset = pandas.read_csv(url, names=names)