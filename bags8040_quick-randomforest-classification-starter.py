# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pulsar_stars = pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')

pulsar_stars
import sklearn

from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.model_selection import GridSearchCV, cross_val_score

import multiprocessing as mps





clf = RFC(n_jobs=mps.cpu_count())

scores = cross_val_score(clf,

                         X=pulsar_stars.drop(axis=1, labels=['target_class']),

                         y=pulsar_stars['target_class'])

scores
if clf is not None:

    del clf

clf = RFC(n_jobs=mps.cpu_count())

params = {

    'criterion':['gini', "entropy"],

    'max_features' : ['auto', 'sqrt', 'log2', None]

}

gs = GridSearchCV(estimator=clf, param_grid=params,

                  scoring='accuracy', cv=4)

gs.fit(X=pulsar_stars.drop(axis=1, labels=['target_class']),

                         y=pulsar_stars['target_class'])

gs.cv_results_
print("Best params:", gs.best_params_)

print("Best scores:", gs.best_score_)

# note 'auto' criterion is same as 'sqrt'
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



tr_X, te_X, tr_y, te_y = train_test_split(pulsar_stars.drop(axis=1, labels=['target_class']),

                                          pulsar_stars['target_class'])

if clf is not None:

    del clf

clf = RFC(n_jobs = mps.cpu_count(),

         criterion = 'entropy',

         max_features = 'auto')

clf.fit(tr_X, tr_y)

y_pred = clf.predict(te_X)

confusion_matrix(y_true=te_y, y_pred=y_pred)
