%matplotlib inline

!pip install yellowbrick
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

from sklearn.dummy import DummyClassifier

import matplotlib.pyplot as plt

from yellowbrick.target import class_balance

from yellowbrick.classifier import ClassificationReport, class_prediction_error, roc_auc



input_path = '/kaggle/input/iris-flower-dataset/IRIS.csv'

iris_df = pd.read_csv(input_path, index_col=False)

X = iris_df.drop('species', axis=1)

y = iris_df['species']

visualizer = class_balance(y)
baseline_clf = DummyClassifier(strategy='stratified', random_state=42)

clf_report = ClassificationReport(baseline_clf, support="count")

clf_report.fit(X, y)

clf_report.score(X, y)

clf_report.finalize()
clf_error = class_prediction_error(baseline_clf, X, y)
auc = roc_auc(baseline_clf, X, y)
baseline_clf.score(X, y)