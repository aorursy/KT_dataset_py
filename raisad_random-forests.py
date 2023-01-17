# -*- coding: utf-8 -*-
"""
Random Forests
"""
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

# read the data in
train = pd.read_csv("train.csv")
train_cols = train.columns[1:369]
y = train['TARGET']

# re-scaling
M = train[train_cols]
M_max = M.max()
M_min = M.min()
ind_cols = (M_max - M_min != 0)
ind_cols = np.where(ind_cols)
X = (M - M_min) / (M_max - M_min)
X = X[train_cols[ind_cols]]

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(X, y)

Z = forest.predict_proba(X)
yhat = Z[0:, 1]

from sklearn.metrics import roc_curve, auc
# Compute micro-average ROC curve and ROC area
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), yhat.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#import test set
test = pd.read_csv("test.csv")

#re-scaling and best features
X_test = (test - M_min) / (M_max - M_min)
X_test = X_test[train_cols[ind_cols]]

#calculate probabilities according to the model
Z_test = forest.predict_proba(X_test)
y_test = Z_test[0:, 1]

#export file
id = np.asarray(test['ID'])
id = id.reshape(id.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
file_export = np.concatenate((id, y_test), axis=1)

np.savetxt("sample_submission.csv", file_export, fmt='%d, %f',  delimiter=",")
