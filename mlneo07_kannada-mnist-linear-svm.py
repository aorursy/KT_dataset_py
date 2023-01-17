import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC

import os
ftrain = "../input/Kannada-MNIST/train.csv"

ftest = "../input/Kannada-MNIST/test.csv"

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
df_train = pd.read_csv(ftrain)

df_test = pd.read_csv(ftest)
df_train.head(10)
df_test.head(10)


X = df_train[df_train.columns[1:]].values

Y = df_train.label.values
#Using linear SVM 

lin_clf = LinearSVC()

lin_clf.fit(X,Y)
preds = lin_clf.predict(df_test[df_test.columns[1:]].values)

submission['label'] = preds

submission.to_csv('submission.csv', index=False)

submission.head(20)