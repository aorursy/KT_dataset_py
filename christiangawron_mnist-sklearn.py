# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import sklearn

from sklearn import svm, metrics



train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv", dtype=np.float32)

test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv", dtype=np.float32)

sample_sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
y = train_df.label.values

X = train_df.loc[:, train_df.columns != 'label'].values/255



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train.shape, X_test.shape
def evaluate(classifier, verbose=False):

    """ Print evaluation data for the classifier """

    predicted = classifier.predict(X_test)

    if verbose:

        print(f"Classification report for classifier {classifier}:")

        print(f"{metrics.classification_report(y_test, predicted)}")

        print(f"Confusion matrix:\n{metrics.confusion_matrix(y_test, predicted)}")

    

    return metrics.f1_score(y_test, predicted, average="weighted")
from sklearn.neural_network import MLPClassifier



classifiers = [ MLPClassifier() ]

scores = {}



for classifier in classifiers:

    print(f"Training: {classifier.__class__.__name__}")

    classifier.fit(X_train, y_train)

    scores[classifier] = evaluate(classifier)

    

for (c, score) in sorted(scores.items(), key=lambda p: p[1], reverse=True):

    print(f"{c.__class__.__name__:25s}: {score:.3f}")



X = test_df.loc[:].values/255



X.shape
classifier = classifiers[0]



predicted = classifier.predict(X).astype(int)



predicted
sample_sub['Label'] = predicted

sample_sub.to_csv("submission.csv", index=False)