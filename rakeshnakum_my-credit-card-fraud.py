# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
y = df['Class']
X = df.drop(columns={'Class'})
X.shape
y.shape
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train.shape
sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train)
## Accuracy on Cross validation ##

from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

y_train_pred = sgd_clf.predict(X_train)
## confusion Matrix ##

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train, y_train_pred)
## precision score ##

from sklearn.metrics import precision_score, recall_score

precision_score(y_train, y_train_pred)
## recall score ##

recall_score(y_train, y_train_pred)
## f1 score ##

from sklearn.metrics import f1_score

f1_score(y_train, y_train_pred)
from sklearn.model_selection import cross_val_predict

y_scores = cross_val_predict(sgd_clf, X_train, y_train, cv=3, method="decision_function")
from sklearn.metrics import precision_recall_curve

precision, recalls, thresholds = precision_recall_curve(y_train, y_scores)
## Precision vs Recall Curve ##

import matplotlib.pyplot as plt

def plot_precision_recall_vs_threshold(precision, recalls, threshols):

    plt.plot(thresholds, precision[:-1], "b--", label="Precision")

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

    [...]

plot_precision_recall_vs_threshold(precision, recalls, thresholds)

plt.show()
## ROC curve ##

from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_train, y_scores)
def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal

    plt.xlabel('False Positive Rate')# Add axis labels and grid

    plt.ylabel("True Positive Rate")

    

plot_roc_curve(fpr, tpr)
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train, y_scores)
from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(random_state = 30)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train, y_scores_forest)
plt.plot(fpr, tpr, "b:", label="SGD")

plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")

plt.legend(loc="lower right")

plt.show()
roc_auc_score(y_train, y_scores_forest)
from sklearn.svm import SVC

svm_clf = SVC()

svm_clf.fit(X_train, y_train)
y_train_pred_svm = svm_clf.predict(X_train)
# from sklearn.metrics import precision_score, recall_score

precision_score(y_train, y_train_pred_svm)
f1_score(y_train, y_train_pred_svm)
y_scores_svm = cross_val_predict(svm_clf, X_train, y_train, cv=3, method="decision_function")
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_train, y_scores_svm)
def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal

    [...] # Add axis labels and grid

    

plot_roc_curve(fpr_svm, tpr_svm)
roc_auc_score(y_train, y_scores_svm)