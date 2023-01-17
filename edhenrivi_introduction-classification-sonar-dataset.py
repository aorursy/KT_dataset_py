# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.read_csv('../input/mines-vs-rocks/sonar.all-data.csv')
main_df = pd.read_csv('/kaggle/input/mines-vs-rocks/sonar.all-data.csv',header=None)

main_df
main_df[60].value_counts().plot(kind='barh')
inputs_df = main_df.drop(60, axis=1)

inputs_df.head()
targets_df = pd.get_dummies(main_df[60])

targets_df
rock_y_df = targets_df['R']

mine_y_df = targets_df['M']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(inputs_df, mine_y_df, test_size=0.30, random_state=42)
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures



# For feature creation

poly = PolynomialFeatures(2)
#Importing classifiers

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier
classifiers_ = [

    ("AdaBoost",AdaBoostClassifier()),

    ("Decision Tree", DecisionTreeClassifier(max_depth=10)),

    ("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))),

    ("Linear SVM", SVC(kernel="linear", C=0.025,probability=True)),

    ("Naive Bayes",GaussianNB()),

    ("Nearest Neighbors",KNeighborsClassifier(3)),

    ("Neural Net",MLPClassifier(alpha=1)),

    ("QDA", QuadraticDiscriminantAnalysis()),

    ("Random Forest",RandomForestClassifier(n_jobs=2, random_state=1)),

    ("RBF SVM",SVC(gamma=2, C=1,probability=True)),

    ("SGDClassifier", SGDClassifier(max_iter=1000, tol=10e-3,penalty='elasticnet'))

    ]
clf_names = []

train_scores = []

test_scores = []

for n,clf in classifiers_:

    clf_names.append(n)

    # Model declaration with pipeline

    clf = Pipeline([('POLY', poly),('CLF',clf)])

    

    # Model training

    clf.fit(X_train, y_train)

    print(n+" training done!")

    

    # Measure training accuracy and score

    train_scores.append(clf.score(X_train, y_train))

    print(n+" training score done!")

    

    # Measure test accuracy and score

    test_scores.append(clf.score(X_test, y_test))

    print(n+" testing score done!")

    print("---")
#Plot results

plt.title('Accuracy Training Score')

plt.grid()

plt.plot(train_scores,clf_names)

plt.show()



plt.title('Accuraccy Test Score')

plt.grid()

plt.plot(test_scores,clf_names)

plt.show()
rng = np.random.RandomState(1)



clf = GaussianProcessClassifier(1.0 * RBF(1.0))



clf = Pipeline([('POLY', poly),

                ('ADABOOST', clf)])



# Training our model

%time clf.fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)
clf.predict(X_test).sum()
y_test.sum()
from sklearn.metrics import plot_confusion_matrix



disp = plot_confusion_matrix(clf, X_train, y_train,

                             display_labels=['ROCK','MINE'],

                             cmap=plt.cm.Blues,

                             normalize=None)

disp.ax_.set_title('Confusion matrix')



print('Train results: confusion matrix')

print(disp.confusion_matrix)
disp = plot_confusion_matrix(clf, X_test, y_test,

                             display_labels=['ROCK','MINE'],

                             cmap=plt.cm.Blues,

                             normalize=None)

disp.ax_.set_title('Confusion matrix')



print('Test results: confusion matrix')

print(disp.confusion_matrix)