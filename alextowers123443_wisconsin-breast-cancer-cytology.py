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
df = pd.read_csv("../input/wisconsin_breast_cancer.csv")

df.fillna(0, inplace=True)

df.head()
import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

x=df.iloc[:, 1:10]

y=df.iloc[:, -1:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1105233)

print(x_train.shape, y_train.shape)

#import SGDClassifier

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=1105233)
#import cross validator

from sklearn.model_selection import cross_val_score
def decode (int, object):

    subcols = list()

    if (subset & 1 == 1):

        subcols.append(x_train.columns[0])

    if (subset & 2 == 2):

        subcols.append(x_train.columns[1])

    if (subset & 4 == 4):

        subcols.append(x_train.columns[2])

    if (subset & 8 == 8):

        subcols.append(x_train.columns[3])

    if (subset & 16 == 16):

        subcols.append(x_train.columns[4])

    if (subset & 32 == 32):

        subcols.append(x_train.columns[5])

    if (subset & 64 == 64):

        subcols.append(x_train.columns[6])

    if (subset & 128 == 128):

        subcols.append(x_train.columns[7])

    if (subset & 256 == 256):

        subcols.append(x_train.columns[8])    

    return subcols;    
all_cv_scores = list()

all_cols = list()



for subset in range (1, 512):

    cols = decode(subset, x_train.columns)   

    cv_score = cross_val_score(sgd_clf, x_train[cols], y_train, cv=10, scoring = "accuracy")    

    all_cv_scores.append(cv_score.mean())

    all_cols.append(cols)
max_accuracy = (max(all_cv_scores))

max_index = all_cv_scores.index(max_accuracy)

print(max(all_cv_scores))

print(all_cv_scores.index(max_accuracy))

print(all_cols[max_index])

from sklearn.metrics import accuracy_score

all_test_scores = list()

sgd_clf_test = SGDClassifier(random_state=1105233)

for i in range (len(all_cols)):

    sgd_clf_test.fit(x_train[all_cols[i]], y_train)

    pred = sgd_clf_test.predict(x_test[all_cols[i]])

    accuracy = accuracy_score(y_test, pred)

    all_test_scores.append(accuracy)
print(all_test_scores[301])

print(max(all_test_scores))
import matplotlib.pyplot as plt



plt.scatter(all_cv_scores, all_test_scores, alpha=0.4)

plt.ylim(ymin=0)

plt.xlabel("CV Accuracy")

plt.ylabel("Test Accuracy")
from sklearn.ensemble import RandomForestClassifier



rf_clf = RandomForestClassifier(n_estimators=30, random_state=1105233)
all_rfcv_scores = list()



for i in range (len(all_cols)):     

    rf_score = cross_val_score(rf_clf, x_train[all_cols[i]], y_train, cv=10, scoring = "accuracy")    

    all_rfcv_scores.append(rf_score.mean())
print(max(all_rfcv_scores))

print(all_rfcv_scores.index(max(all_rfcv_scores)))
all_rf_test_scores = list()

for i in range (len(all_cols)):

    rf_clf.fit(x_train[all_cols[i]], y_train)

    rf_pred = rf_clf.predict(x_test[all_cols[i]])

    rf_accuracy = accuracy_score(y_test, rf_pred)

    all_rf_test_scores.append(rf_accuracy)
print(all_rf_test_scores[110])

print(max(all_rf_test_scores))

print(all_rf_test_scores.index(max(all_rf_test_scores)))

print(all_cols[110])

print(all_cols[164])
plt.scatter(all_rfcv_scores, all_rf_test_scores, alpha=0.4)

plt.ylim(ymin=0)

plt.xlabel("CV Accuracy")

plt.ylabel("Test Accuracy")
from sklearn.naive_bayes import GaussianNB

g_clf = GaussianNB()
all_gcv_scores = list()



for i in range (len(all_cols)):     

    g_score = cross_val_score(g_clf, x_train[all_cols[i]], y_train, cv=10, scoring = "accuracy")    

    all_gcv_scores.append(g_score.mean())
print(max(all_gcv_scores))

print(all_gcv_scores.index(max(all_gcv_scores)))
all_g_test_scores = list()

for i in range (len(all_cols)):

    g_clf.fit(x_train[all_cols[i]], y_train)

    g_pred = g_clf.predict(x_test[all_cols[i]])

    g_accuracy = accuracy_score(y_test, g_pred)

    all_g_test_scores.append(g_accuracy)
print(all_g_test_scores[34])

print(max(all_g_test_scores))

print(all_g_test_scores.index(max(all_g_test_scores)))

print(all_cols[34])

print(all_cols[146])
plt.scatter(all_gcv_scores, all_g_test_scores, alpha=0.4)

plt.ylim(ymin=0)

plt.xlabel("CV Accuracy")

plt.ylabel("Test Accuracy")
from sklearn.model_selection import cross_val_predict



sgd_clf_final = SGDClassifier(random_state=1105233)

rf_clf_final = RandomForestClassifier(n_estimators=30, random_state=1105233)

g_clf_final = GaussianNB()



sgd_pred = cross_val_predict(sgd_clf_final, x, y, cv=10, method='decision_function')

rf_pred = cross_val_predict(rf_clf_final, x, y, cv=10, method='predict_proba')

g_pred = cross_val_predict(g_clf_final, x, y, cv=10, method='predict_proba')
#method to plot roc curves

def plot_roc_curve(fpr,tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    

    plt.plot([0,1], [0,1], 'k--')
from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y, sgd_pred)

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y, rf_pred[:, 1])

fpr_gauss, tpr_gauss, thresholds_gauss = roc_curve(y, g_pred[:, 1])

print(thresholds)

print(thresholds_forest)

print(thresholds_gauss)





plot_roc_curve(fpr, tpr, "SGD")

plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")

plot_roc_curve(fpr_gauss, tpr_gauss, "GaussianNB")

plt.legend()

plt.xlabel("FPR")

plt.ylabel("TPR")