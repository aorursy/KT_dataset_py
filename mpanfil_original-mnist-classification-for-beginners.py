import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')

x = mnist.data

y = mnist.target
import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
digit = x[36000]

digit_img = digit.reshape(28,28)



plt.imshow(digit_img, cmap=matplotlib.cm.binary, interpolation='nearest')

plt.axis('off')

plt.show()
y[36000]
# Splitting train and test data



X_train, X_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
# Shuffling dataset



shuffle_index = np.random.permutation(60000)

X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
# Binary classifier for '9' digit



y_train_9 = (y_train == '9')

y_test_9 = (y_test == '9')
np.unique(y_train_9)
# SGD Classifier



from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_9)
sgd_clf.predict([digit])
from sklearn.model_selection import cross_val_score



cross_val_score(sgd_clf, X_train, y_train_9, cv=3, scoring='accuracy')
from sklearn.model_selection import cross_val_predict



y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_9, cv=3)
from sklearn.metrics import confusion_matrix



confusion_matrix(y_train_9, y_train_pred)
from sklearn.metrics import precision_score, recall_score



precision_score(y_train_9, y_train_pred)
recall_score(y_train_9, y_train_pred)
from sklearn.metrics import f1_score



f1_score(y_train_5, y_train_pred)
y_scores = sgd_clf.decision_function([digit])

y_scores

threshold = 0

y_some_digit_pred = (y_scores >threshold)

y_some_digit_pred
# How to set threshold for decision function?
# Results for every examples of train set

y_scores = cross_val_predict(sgd_clf, X_train, y_train_9, cv=3, method='decision_function')

y_scores
from sklearn.metrics import precision_recall_curve



precisions, recalls, thresholds = precision_recall_curve(y_train_9, y_scores)
def plot_precision_recall_vs_th(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')

    plt.plot(thresholds, recalls[:-1], 'g--', label='Recall')

    plt.xlabel('Threshold')

    plt.legend(loc='center left')

    plt.ylim([0,1])
plot_precision_recall_vs_th(precisions, recalls, thresholds)

plt.figure(figsize=(20,10))

plt.show()
plt.plot(recalls[:-1], precisions[:-1])

plt.xlabel('Recall')

plt.ylabel('Precision')
import warnings

warnings.filterwarnings('always')
# Precision - 90% -> threshold = 70 000



y_train_pred_90 = (y_scores > 70000)

precision_score(y_train_9, y_train_pred_90, zero_division=1)

recall_score(y_train_9, y_train_pred_90, zero_division=1)
# ROC curve



from sklearn.metrics import roc_curve, roc_auc_score



fpr, tpr, thresholds = roc_curve(y_train_9, y_scores)
def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0,1], [0,1], 'k--')

    plt.axis([0,1,0,1])

    plt.xlabel('False positive')

    plt.ylabel('True positive')



plot_roc_curve(fpr, tpr)

plt.show()
# Area under the curve - AUC score

roc_auc_score(y_train_9, y_scores)
from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_9, cv=3, method='predict_proba')
y_probas_forest
y_scores_forest = y_probas_forest[:, 1] #result = probability to positive class

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_9, y_scores_forest)
plt.plot(fpr, tpr, 'b:', label='SGD')

plot_roc_curve(fpr_forest, tpr_forest, 'Random Forest')

plt.legend(loc='lower right')

plt.show()
roc_auc_score(y_train_9, y_scores_forest)