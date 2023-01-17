# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
%ls ../input
df = pd.read_csv("../input/Iris.csv", index_col=None, sep=',')
df.head()
# from here onwards, code heavily inspired by Raul Garreta's book 'Learning scikit learn: machinelearning in Python'
df.Species.unique()

df['Species_cat'] = pd.Categorical(df['Species'], categories=df['Species'].unique()).codes

df.head()                     
X_iris, y_iris = df.iloc[:,1:5], df['Species_cat']

print(X_iris.shape, y_iris.shape)
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

X,y  = X_iris.iloc[:, :2] , y_iris
X.head()
# Make a train and test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train.shape, y_train.shape
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
import matplotlib.pyplot as plt
colours = ['red', 'greenyellow', 'blue']

for i in range(len(colours)):
    xs = X_train.iloc[:, 0][y_train == i]
    ys = X_train.iloc[:, 1][y_train == i]
    
    plt.scatter(xs, ys, c=colours[i])
    
plt.legend(df.Species.unique())    
    
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

from sklearn.linear_model import SGDClassifier

clf = SGDClassifier()
clf.fit(X_train_scaled, y_train)

print(clf.coef_)

print(clf.intercept_)
x_min, x_max = X_train_scaled[:,0].min() - 0.5, X_train_scaled[:,0].max() + 0.5

y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5

xs = np.arange(x_min, x_max, 0.5)
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 6)

for i in [0, 1, 2]:
    
    axes[i].set_aspect('equal')
    axes[i].set_title('Class ' + str(i) + ' versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    plt.sca(axes[i])
    
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:,1], c=y_train, cmap=plt.cm.prism)

    ys = (- clf.intercept_[i]   - xs * clf.coef_[i, 0]) / clf.coef_[i, 1]

    plt.plot(xs, ys)
    
    
    
print (clf.predict(scaler.transform([[4.7, 3.1]])))
print(clf.decision_function(scaler.transform([[4.7, 3.1]])))
from sklearn import metrics

y_train_pred = clf.predict(X_train_scaled)

print ('Accuracy on training set: ', metrics.accuracy_score(y_train, y_train_pred))
y_pred = clf.predict(X_test_scaled)
print ('Accuracy on testing set: ', metrics.accuracy_score(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred, target_names=df.Species.unique()))
