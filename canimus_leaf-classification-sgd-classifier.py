%matplotlib inline

import matplotlib

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pylab as plt

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
set(train.columns) - set(test.columns)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder().fit(train.species) 

labels = label_encoder.transform(train.species)

classes = list(label_encoder.classes_) 
train = train.drop(['species', 'id'], axis=1)

test_ids = test.id

test = test.drop(['id'], axis=1)
# Shuffling data set with guarantee of labels across split



from sklearn.cross_validation import StratifiedShuffleSplit

split_shuffle = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)



for train_index, test_index in split_shuffle:

    X_train, X_cross = train.values[train_index], train.values[test_index]

    y_train, y_cross = labels[train_index], labels[test_index]
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, log_loss



clf = SGDClassifier(alpha=0.000001, average=False, class_weight=None, epsilon=0.1,

       eta0=0.0, fit_intercept=True, l1_ratio=0.10,

       learning_rate='optimal', loss='log', n_iter=1000, n_jobs=10,

       penalty='l2', power_t=0.41, random_state=None, shuffle=True,

       verbose=0, warm_start=False)



# Fitting the model

clf.fit(X_train, y_train)
train_predictions = clf.predict(X_cross)

acc = accuracy_score(y_cross, train_predictions)

print("Accuracy: {:.4%}".format(acc))



train_predictions = clf.predict_proba(X_cross)

ll = log_loss(y_cross, train_predictions)

print("Log Loss: {}".format(ll))
# Representation of dataset

plt.figure(figsize=(12,6))

ax = sns.heatmap(X_train, xticklabels=False, yticklabels=False, cbar=False)
test_predictions = clf.predict_proba(test)



# Format DataFrame

submission = pd.DataFrame(test_predictions, columns=classes)

submission.insert(0, 'id', test_ids)

submission.reset_index()



# Export Submission

submission.to_csv('submission.csv', index = False)

submission.tail()