# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
data = pd.read_csv("../input/data.csv",header=0)
# Any results you write to the current directory are saved as output.
print(data.columns)
print(data.shape)
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
print(data.columns)  # To check if columns are dropped
print(data['diagnosis'].value_counts())
def accuracy_predictor(model, data):
    train = data.drop('diagnosis', axis=1)
    label = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(train,
                                                        label,
                                                        test_size=0.3,
                                                        random_state=0)
    clf = model.fit(X_train, y_train)
    print("Using all features %f" % clf.score(X_test, y_test))
clf = svm.SVC(kernel='linear', C=1)
accuracy_predictor(clf, data)
import matplotlib.pyplot as plt
import seaborn as sns
# I will make use of all features which are labelled as *.mean 
features_mean = ['radius_mean', 'texture_mean', 'perimeter_mean',
                 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
corr = data[features_mean].corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 8},
           xticklabels= features_mean, yticklabels= features_mean,
           cmap= 'coolwarm') 
plt.show()

def accuracy_predictor(model, data):
    train = data.drop('diagnosis', axis=1)
    label = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(train,
                                                        label,
                                                        test_size=0.3,
                                                        random_state=0)
    clf = model.fit(X_train, y_train)
    print("Using all features %f" % clf.score(X_test, y_test))
    unique_mean_features = ['radius_mean', 'texture_mean', 'smoothness_mean',
                            'compactness_mean', 'symmetry_mean',
                            'fractal_dimension_mean']
    new_train = data[unique_mean_features]
    X_train, X_test, y_train, y_test = train_test_split(new_train,
                                                        label,
                                                        test_size=0.3,
                                                        random_state=0)
    clf = model.fit(X_train, y_train)
    print("With independent mean features: %f" % clf.score(X_test, y_test))

clf = svm.SVC(kernel='linear', C=1)
accuracy_predictor(clf, data)
clf = tree.DecisionTreeClassifier()
accuracy_predictor(clf, data)
knn = KNeighborsClassifier(n_neighbors=5,
                           algorithm='ball_tree'
                          )
accuracy_predictor(knn, data)
for neighbor in range(3, 20):
    print("Iteration %d" % neighbor)
    knn = KNeighborsClassifier(n_neighbors=neighbor,
                               algorithm='ball_tree'
                               )
    accuracy_predictor(knn, data)
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

x = range(3, 20)
# y1 and y2 are scores when using all features and unique features respectively.
# I made slight changes in the function and appended the scores in two lists.
# I am not posting that basic code.
y1 = [0.9181286549707602, 0.9298245614035088, 0.9473684210526315, 0.9473684210526315, 0.9532163742690059, 0.9532163742690059,
      0.9590643274853801, 0.9649122807017544, 0.9649122807017544, 0.9649122807017544,
      0.9649122807017544, 0.9649122807017544, 0.9649122807017544, 0.9649122807017544,
      0.9649122807017544, 0.9649122807017544, 0.9649122807017544]
y2 = [0.8421052631578947,
      0.8596491228070176, 0.8771929824561403, 0.8830409356725146, 0.8888888888888888,
      0.9005847953216374, 0.9064327485380117, 0.8888888888888888, 0.8947368421052632,
      0.8947368421052632, 0.8947368421052632, 0.9005847953216374, 0.8947368421052632,
      0.8947368421052632, 0.9005847953216374, 0.9005847953216374, 0.9064327485380117]
f1 = interp1d(x, y1, kind='cubic')
f2 = interp1d(x, y2, kind='cubic')
plt.plot(x, f1(x), '-', x, f2(x), '--')
plt.legend(['all features', 'unique mean features'], loc='best')
plt.show()