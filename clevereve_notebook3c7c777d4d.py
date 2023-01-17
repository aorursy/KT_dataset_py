# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedShuffleSplit

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# Any results you write to the current directory are saved as output.
def encode(train, test):

	le = LabelEncoder().fit(train.species) 

	labels = le.transform(train.species)           # encode species strings

	classes = list(le.classes_)                    # save column names for submission

	test_ids = test.id                             # save test ids for submission

    

	train = train.drop(['species', 'id'], axis=1)  

	test = test.drop(['id'], axis=1)

    

	return train, labels, test, test_ids, classes



train, labels, test, test_ids, classes = encode(train, test)



train.head(1)
sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

for train_index, test_index in sss:

	X_train, X_test = train.values[train_index], train.values[test_index]

	y_train, y_test = labels[train_index], labels[test_index]
from sklearn.metrics import accuracy_score, log_loss

from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import CalibratedClassifierCV as cc
clf = RandomForestClassifier(n_estimators=1000)

clf = cc(clf, cv=3, method='isotonic')

clf.fit(train, labels)
predictions = clf.predict_proba(test)

np.shape(predictions)
sub = pd.DataFrame(predictions, columns=classes)

sub.insert(0, 'id', test_ids)

sub.reset_index()

sub.to_csv('submit.csv', index = False)

sub.head() 