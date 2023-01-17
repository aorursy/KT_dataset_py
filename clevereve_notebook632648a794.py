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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cross_validation import StratifiedKFold

from sklearn.linear_model import LogisticRegression



def warn(*args, **kwargs): pass

import warnings

warnings.warn = warn



from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedShuffleSplit



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



def encode(train, test):

	le = LabelEncoder().fit(train.species) 

	labels = le.transform(train.species)           # encode species strings

	classes = list(le.classes_)                    # save column names for submission

	test_ids = test.id                             # save test ids for submission

    

	train = train.drop(['species', 'id'], axis=1)  

	test = test.drop(['id'], axis=1)

    

	return train, labels, test, test_ids, classes



train, labels, test, test_ids, classes = encode(train, test)



from sklearn.metrics import accuracy_score, log_loss

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier



n_folds = 5



skf = list(StratifiedKFold(labels, n_folds))





clfs = [

    RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),

    RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),

    ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),

    ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),

    GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),

]



dataset_blend_train = np.zeros((train.shape[0], len(clfs)))

dataset_blend_test = np.zeros((test.shape[0], len(clfs)))



print(np.shape(train))

for j, clf in enumerate(clfs):

    dataset_blend_test_j = np.zeros((test.shape[0], len(skf)))

   

    for i, (idx_train, idx_test) in enumerate(skf):

        X_train = train.loc[idx_train]

        y_train = labels[idx_train]

        X_test = train.loc[idx_test]

        y_test = labels[idx_test]

        clf.fit(X_train, y_train)

       

        y_submission=clf.predict_proba(X_test)

        

        dataset_blend_train[idx_test, j] = y_submission[:,1]

        dataset_blend_test_j[:,i]=clf.predict_proba(test)[:,1]

    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    

import xgboost 

clf = xgboost()

clf.fit(dataset_blend_train, labels)

predictions = clf.predict_proba(dataset_blend_test)



sub = pd.DataFrame(predictions, columns=classes)

sub.insert(0, 'id', test_ids)

sub.reset_index()

sub.to_csv('submit.csv', index = False)

sub.head()