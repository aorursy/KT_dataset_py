# Check the versions of libraries

# Python version

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
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"

#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

#dataset = pandas.read_csv(url, names=names)

data='../input/iris-flower-dataset/IRIS.csv'

#reading dataset

dataset=pandas.read_csv(data)
print('shape:')

print(dataset.shape)    #outputs shape of the dataset

print('data:')

print(dataset.head())    #prints top 5 data points

print('description:')

print(dataset.describe())    #computs mean ,std ,min,max etc

print('number of examples:')

print(dataset.groupby('species').size())   #class distribution
#plotting box plot

dataset.plot(kind='box',subplots=True, layout=(2,2),sharex=False, sharey=False)

plt.show()

#histogram

dataset.hist()

plt.show()
# scatter plot matrix

scatter_matrix(dataset)

plt.show()
array=dataset.values

X=array[:,0:4]

Y=array[:,4]

validation_size=0.2

seed=1

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

seed=1

scoring='accuracy'
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
# Make predictions on validation dataset

knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

predictions = knn.predict(X_validation)

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))