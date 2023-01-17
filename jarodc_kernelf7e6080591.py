# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

from sklearn.model_selection import train_test_split

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

import numpy as np

import pandas as pd

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline

from collections import Counter

from sklearn.linear_model import LogisticRegression

import time



from sklearn import datasets

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report



train=pd.read_csv('../input/cat-in-the-dat/train.csv', delimiter=',') 

test = pd.read_csv('../input/cat-in-the-dat/test.csv', delimiter=',') 

id = test['id']

test = test.drop(columns="id")



bin3_map = {'T': 1, 'F': -1}



bin4_map = {'Y': 1, 'N': -1}



ord1_map = {'Novice': 1, 'Contributor': 2,

               'Expert': 3, 'Master': 4, 'Grandmaster': 5}



ord2_map = {'Freezing': 1, 'Cold': 2,

               'Warm': 3, 'Hot': 4, 'Boiling Hot': 5, 'Lava Hot': 6}



ord3_map = {'a': 1, 'b': 2,

               'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,

               'i': 9, 'j': 10, 'k': 11, 'l': 12,'m': 13, 'n': 14, 'o': 15}



ord4_map = {'A': 1, 'B': 2,

               'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,

               'I': 9, 'J': 10, 'K': 11, 'L': 12,'M': 13, 'N': 14, 'O': 15,'P': 16, 'Q': 17,

               'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23,

               'X': 24, 'Y': 25, 'Z': 26}



train['bin_3'] = train['bin_3'].map(bin3_map)

train['bin_4'] = train['bin_4'].map(bin4_map)

train['ord_1'] = train['ord_1'].map(ord1_map)

train['ord_2'] = train['ord_2'].map(ord2_map)

train['ord_3'] = train['ord_3'].map(ord3_map)

train['ord_4'] = train['ord_4'].map(ord4_map)



train['dy_sin'] = np.sin((train['day']-1)*(2.*np.pi/7))

train['dy_cos'] = np.cos((train['day']-1)*(2.*np.pi/7))

train['mnth_sin'] = np.sin((train['month']-1)*(2.*np.pi/12))

train['mnth_cos'] = np.cos((train['month']-1)*(2.*np.pi/12))



train = train.drop(columns="day")

train = train.drop(columns="month")

train = train.drop(columns="ord_5")

train = train.drop(columns="nom_5")

train = train.drop(columns="nom_6")

train = train.drop(columns="nom_7")

train = train.drop(columns="nom_8")

train = train.drop(columns="nom_9")

train = train.drop(columns="id")

target=train['target']

train = train.drop(columns="target")

train['target']=target

test['bin_3'] = test['bin_3'].map(bin3_map)

test['bin_4'] = test['bin_4'].map(bin4_map)

test['ord_1'] = test['ord_1'].map(ord1_map)

test['ord_2'] = test['ord_2'].map(ord2_map)

test['ord_3'] = test['ord_3'].map(ord3_map)

test['ord_4'] = test['ord_4'].map(ord4_map)



test['dy_sin'] = np.sin((test['day']-1)*(2.*np.pi/7))

test['dy_cos'] = np.cos((test['day']-1)*(2.*np.pi/7))

test['mnth_sin'] = np.sin((test['month']-1)*(2.*np.pi/12))

test['mnth_cos'] = np.cos((test['month']-1)*(2.*np.pi/12))



test = test.drop(columns="day")

test = test.drop(columns="month")

test = test.drop(columns="ord_5")

test = test.drop(columns="nom_5")

test = test.drop(columns="nom_6")

test = test.drop(columns="nom_7")

test = test.drop(columns="nom_8")

test = test.drop(columns="nom_9")
ohn=OneHotEncoder(sparse=False)

column_trans=make_column_transformer((OneHotEncoder(),['nom_0','nom_1',

'nom_2','nom_3','nom_4']),remainder='passthrough')



test = column_trans.fit_transform(test)



data=column_trans.fit_transform(train)

n = data.shape[0]

m=int(0.8*n)



X_train = data[:,0:39]

y_train = data[:,39]



"""

train=data[:m,:]

test=data[m:-1,:]



X_train=train[:,0:39]

y_train=train[:,39]



X_test=test[:,0:39]

y_test=test[:,39]

"""
"""

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",

          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",

          "Naive Bayes", "QDA"]







classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    MLPClassifier(alpha=1, max_iter=1000),

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()]





names = ["Decision Tree"]

classifiers = [ DecisionTreeClassifier]



depth_range = 9



tuned_parameters_decisiontree = dict(max_depth = depth_range)

print(tuned_parameters_decisiontree)





clf = GridSearchCV(

         DecisionTreeClassifier(), tuned_parameters_decisiontree)

"""

clf = DecisionTreeClassifier(max_depth = 9)

clf.fit(X_train, y_train)

"""

print("Best parameters set found on development set:")

print()

print(clf.best_params_)

print()

print("Grid scores on development set:")

print()

means = clf.cv_results_['mean_test_score']

stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):

    print("%0.3f (+/-%0.03f) for %r"

        % (mean, std * 2, params))

print()

print("Detailed classification report:")

print()

print("The model is trained on the full development set.")

print("The scores are computed on the full evaluation set.")

print()

y_true, y_pred = y_test, clf.predict(X_test)

print(classification_report(y_true, y_pred))

print()

"""



"""

tree_range = list(range(9,16))

feature_range = list(range(9,16))

depth_range_forest = (list(range(9,16)))

tuned_parameters_randomforest = dict(n_estimators = tree_range, max_depth = depth_range_forest, max_features = feature_range)



clf2 = GridSearchCV(

         RandomForestClassifier(), tuned_parameters_randomforest)



clf2.fit(X_train, y_train)



print("Best parameters set found on development set:")

print()

print(clf2.best_params_)

print()

print("Grid scores on development set:")

print()

means = clf2.cv_results_['mean_test_score']

stds = clf2.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf2.cv_results_['params']):

    print("%0.3f (+/-%0.03f) for %r"

        % (mean, std * 2, params))

print()

print("Detailed classification report:")

print()

print("The model is trained on the full development set.")

print("The scores are computed on the full evaluation set.")

print()

y_true, y_pred = y_test, clf2.predict(X_test)

print(classification_report(y_true, y_pred))

print()

"""
results = clf.predict_proba(test)[:,1]



print (results)



submission = pd.DataFrame({'id': id, 'target': results})

print(submission)

submission.to_csv('submission.csv', index=False)
