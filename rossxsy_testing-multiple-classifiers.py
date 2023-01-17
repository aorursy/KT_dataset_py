import numpy as np

import scipy as sp

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



def warn(*args, **kwargs): pass

import warnings

warnings.warn = warn



%matplotlib inline



# from sklearn.preprocessing import LabelEncoder

# from sklearn.cross_validation import StratifiedShuffleSplit



train = pd.read_csv('../input/train.csv').drop('id',axis=1)

test = pd.read_csv('../input/test.csv')

test_ids = test['id']

test.drop('id',axis=1,inplace=True)
print(train.isnull().any().any())

print(test.isnull().any().any())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
species = train['species']

train.drop('species',axis=1,inplace=True)

y_train = le.fit_transform(species)
from sklearn.preprocessing import MaxAbsScaler
x_data = np.vstack([train,test])

mas = MaxAbsScaler()

n_x_data = mas.fit_transform(x_data)

print(n_x_data.shape)

n_x_data
from sklearn.model_selection import StratifiedShuffleSplit



n_x_test = n_x_data[len(species):,:]

x_train = n_x_data[0:len(species),:]



# val_size = 0.1

# seed = 0

# n_x_train, n_x_val, y_train, y_val = cross_validation.train_test_split(n_x_train, y_train, test_size=val_size, 

#                                                                        random_state=seed, stratify=y_train)



sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)



for train_index, test_index in sss.split(x_train,y_train):

    n_x_train, n_x_val = x_train[train_index], x_train[test_index]

    n_y_train, n_y_val = y_train[train_index], y_train[test_index]
print(n_x_train.shape)

print(n_y_train.shape)

print(n_x_val.shape)

print(n_y_val.shape)

print(n_x_test.shape)
np.isnan(y_train).any()
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import accuracy_score, log_loss
seed=1

models = [

            'ADB',

            'GBC',

            'RFC',

            'KNC',

            'SVC',

            'logisticRegression'

         ]

clfs = [

        AdaBoostClassifier(random_state=seed),

        GradientBoostingClassifier(random_state=seed),

        RandomForestClassifier(random_state=seed,n_jobs=-1),

        KNeighborsClassifier(n_jobs=-1),

        SVC(random_state=seed,probability=True),

        LogisticRegression(solver='newton-cg', multi_class='multinomial')

        ]
params = {

            models[0]:{'learning_rate':[0.01], 'n_estimators':[150]},

            models[1]:{'learning_rate':[0.01],'n_estimators':[100], 'max_depth':[3],

                       'min_samples_split':[2],'min_samples_leaf': [2]},

            models[2]:{'n_estimators':[100], 'criterion':['gini'],'min_samples_split':[2],

                      'min_samples_leaf': [4]},

            models[3]:{'n_neighbors':[5], 'weights':['distance'],'leaf_size':[15]},

            models[4]: {'C':[100], 'tol': [0.005],

                       'kernel':['sigmoid']},

            models[5]: {'C':[2000], 'tol': [0.0001]}

         }
y_test = 0

test_scores = []
for name, estimator in zip(models,clfs):

    print(name)

    clf = GridSearchCV(estimator, params[name], scoring='log_loss', refit='True', n_jobs=-1, cv=5)

    clf.fit(n_x_train, n_y_train)



    print("best params: " + str(clf.best_params_))

    print("best scores: " + str(clf.best_score_))

    estimates = clf.predict_proba(n_x_test)

    y_test+=estimates

    acc = accuracy_score(n_y_val, clf.predict(n_x_val))

    print("Accuracy: {:.4%}".format(acc))

    

    test_scores.append((acc,clf.best_score_))

    

    submission = pd.DataFrame(estimates, index=test_ids, columns=le.classes_)

    submission.to_csv('./'+name+'.csv')
y_test = y_test/len(models)

submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)

submission.to_csv('./avgEnsembles.csv')