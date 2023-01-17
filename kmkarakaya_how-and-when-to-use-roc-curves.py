# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# precision-recall curve and f1

from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from sklearn.metrics import average_precision_score

from matplotlib import pyplot

# generate 2 class dataset

X, y = make_classification(n_samples=1000, n_classes=2, weights=[1,1], random_state=1)

# split into train/test sets

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)

# fit a model

model = KNeighborsClassifier(n_neighbors=3)

model.fit(trainX, trainy)

# predict probabilities

probs_orig = model.predict_proba(testX)



probs
# keep probabilities for the positive (1) outcome only

probs = probs_orig[:, 1]



probs

# predict class values

yhat = model.predict(testX)

yhat
# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(testy, probs)



print('precision: ', precision)

print('recall   :' , recall)

print('thresholds:', thresholds)



# calculate F1 score

f1 = f1_score(testy, yhat)

f1
# calculate precision-recall AUC

auc = auc(recall, precision)

auc
# calculate average precision score

ap = average_precision_score(testy, probs)

print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))

# plot no skill

pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

pyplot.plot(recall, precision, marker='.')

# show the plot

pyplot.show()