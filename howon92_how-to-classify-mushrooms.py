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
data = pd.read_csv('../input/mushrooms.csv')

data.head()
data.columns
Y = data['class']

X = data[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',

       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',

       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',

       'stalk-surface-below-ring', 'stalk-color-above-ring',

       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',

       'ring-type', 'spore-print-color', 'population', 'habitat']]

Y.value_counts()
X.head()
X['cap-shape'].head()
X['cap-shape'].astype('category').head()
X['cap-shape'].value_counts()
categories = X['cap-shape'].astype('category')

categories.head()
categories.cat.codes.head()
for column in X:

    X[column] = X[column].astype('category').cat.codes

X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train.head()
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



h = .02  # step size in the mesh



names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",

         "Decision Tree", "Random Forest"]



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)]

from sklearn.model_selection import cross_val_score



best_clf = {

    'name': None,

    'model': None,

    'score': 0

}



for name, clf in zip(names, classifiers):

    scores = cross_val_score(clf, X_train, Y_train, cv=5)

    score = scores.mean()

    if score > best_clf['score']:

        best_clf['model'] = clf

        best_clf['score'] = score

        best_clf['name'] = name

    print(name, score)

    

print("Best clf:", best_clf['name'], best_clf['score'])
clf = best_clf['model']

clf.fit(X_train, Y_train)

score = clf.score(X_test, Y_test)

print(best_clf['name'], score)