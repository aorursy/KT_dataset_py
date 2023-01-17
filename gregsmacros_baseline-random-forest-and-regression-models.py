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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
buildset = pd.read_csv(r'../input/JH_augmented_tourney.csv')

# remove unnamed col
buildset = buildset.iloc[:, 1:]
# pull numerical chars out
features = list(buildset)[11:19] + list(buildset)[21:]
X = buildset.loc[:, features]
y = buildset.target

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
test_predict = clf.predict_proba(X_test)[:,0]
train_predict = clf.predict_proba(X_train)[:,0]
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, test_predict, pos_label = 0)
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, train_predict, pos_label = 0)
plt.figure()
lw = 2
plt.plot(train_fpr, train_tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc(train_fpr, train_tpr))
plt.plot(test_fpr, test_tpr, color='blue',
         lw=lw, label='ROC curve (area = %0.2f)' % auc(test_fpr, test_tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()