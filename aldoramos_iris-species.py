import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
iris = pd.read_csv('../input/Iris.csv')
iris.info()
iris.head()
# show the class distribution
iris.Species.value_counts()
iris.describe()
fig = plt.figure(figsize=(15, 10))
ax = fig.gca()
iris.hist(ax=ax)
plt.show()
# help function
def custom_scatter(dims, ax):
    x, y = dims
    iris.loc[iris.Species=='Iris-setosa'].plot.scatter(x, y, color='r', label='Setosa', ax=ax)
    iris.loc[iris.Species=='Iris-versicolor'].plot.scatter(x, y, color='g', label='Versicolor', ax=ax)
    iris.loc[iris.Species=='Iris-virginica'].plot.scatter(x, y, color='b', label='Virginica', ax=ax)

# visualizing the relationship between the variables
coor_by_dims = {('SepalLengthCm', 'SepalWidthCm'): (0, 0), 
                ('SepalLengthCm', 'PetalLengthCm'): (0, 1), 
                ('SepalLengthCm', 'PetalWidthCm'): (1, 0),
                ('PetalLengthCm', 'PetalWidthCm'): (1, 1),
                ('PetalLengthCm', 'SepalWidthCm'): (2, 0),
                ('PetalWidthCm', 'SepalWidthCm'): (2, 1)}
_, ax = plt.subplots(3, 2, figsize=(15,15))
for dims, coor in coor_by_dims.items():
    custom_scatter(dims, ax[coor])
plt.show()
# remove unnecessary column
iris = iris.drop(columns='Id')
# split the data into random train and test subsets
X = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# trains a model and computes its accuracy
def make_classifier(clf_class):
    clf_by_dims = {}
    for dims in coor_by_dims.keys():
        x_dim, y_dim = dims
        X_train_temp = X_train[[x_dim, y_dim]]
        X_test_temp = X_test[[x_dim, y_dim]]
        clf = clf_class()
        clf.fit(X_train_temp, y_train)
        y_pred = clf.predict(X_test_temp)
        score = accuracy_score(y_test, y_pred)
        clf_by_dims[dims] = (clf, score)
    return clf_by_dims
# helper function
def plot_decision_boundary(clf, score, dims, ax):
    x_dim, y_dim = dims
    extra = .3
    x_min, x_max = iris[x_dim].min() - extra, iris[x_dim].max() + extra
    y_min, y_max = iris[y_dim].min() - extra, iris[y_dim].max() + extra
    step = .02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    decs_bound = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    class_map = {'Iris-setosa':0, 'Iris-versicolor':1 ,'Iris-virginica':2}
    decs_bound = np.array([class_map[y] for y in decs_bound])
    decs_bound = decs_bound.reshape(xx.shape)

    plt.contourf(xx, yy, decs_bound, alpha=.3)
    plt.title(f'Score: {score}')
    
    ax = plt.gca()
    custom_scatter(dims, ax)
# helper function
def plot_clf_accuracy(clf_by_dims):
    i = 1
    plt.subplots(figsize=(15,15))
    for dims, coor in coor_by_dims.items():
        clf, score = clf_by_dims[dims] 
        ax = plt.subplot(3, 2, i)
        plot_decision_boundary(clf, score, dims, ax)    
        i += 1    
    plt.show()
# train a model using logistic regression with all variables
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
clf_by_dims = make_classifier(LogisticRegression)
plot_clf_accuracy(clf_by_dims)
# train a model using decision tree with all variables
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
clf_by_dims = make_classifier(DecisionTreeClassifier)
plot_clf_accuracy(clf_by_dims)
# plot feature importances
plt.figure(figsize=(7, 5))
plt.barh(X_train.columns, clf.feature_importances_)
plt.show()
# train a model using random forest with all variables
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
clf_by_dims = make_classifier(RandomForestClassifier)
plot_clf_accuracy(clf_by_dims)
# plot feature importances
plt.figure(figsize=(7, 5))
plt.barh(X_train.columns, clf.feature_importances_)
plt.show()