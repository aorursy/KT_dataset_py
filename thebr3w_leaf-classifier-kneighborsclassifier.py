import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



def warn(*args, **kwargs): pass

import warnings

warnings.warn = warn



from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.cross_validation import train_test_split



from sklearn.metrics import accuracy_score, log_loss



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

train.head(1)
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
sss = StratifiedShuffleSplit(labels, 10, test_size=0.1, random_state=23)



for train_index, test_index in sss:

    X_train, X_test = train.values[train_index], train.values[test_index]

    y_train, y_test = labels[train_index], labels[test_index]

    

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)



X_test = scaler.transform(X_test)
clf = KNeighborsClassifier(leaf_size=10,n_neighbors=1, metric='manhattan')

clf.fit(X_train, y_train)



# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)



name = clf.__class__.__name__



print("="*30)

print(name)



print('****Results****')

train_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, train_predictions)

print("Accuracy: {:.4%}".format(acc))



train_predictions = clf.predict_proba(X_test)

ll = log_loss(y_test, train_predictions)

print("Log Loss: {}".format(ll))    



log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

log = log.append(log_entry)

print(log)    

print("="*30)