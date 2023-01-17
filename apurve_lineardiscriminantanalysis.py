import numpy as np

import pandas as pd

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
sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)



for train_index, test_index in sss:

    X_train, X_test = train.values[train_index], train.values[test_index]

    y_train, y_test = labels[train_index], labels[test_index]
from sklearn.metrics import accuracy_score, log_loss

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



clf = LinearDiscriminantAnalysis()

clf.fit(X_train, y_train)

    

train_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, train_predictions)

print("Accuracy: {:.4%}".format(acc))

    

train_predictions = clf.predict_proba(X_test)

ll = log_loss(y_test, train_predictions)

print("Log Loss: {}".format(ll))
test_predictions = clf.predict_proba(test)



submission = pd.DataFrame(test_predictions, columns=classes)

submission.insert(0, 'id', test_ids)

#submission1.reset_index()

submission.to_csv('submission.csv')

submission.tail()