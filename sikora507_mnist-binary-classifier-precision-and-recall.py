# import all common modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib
import matplotlib.pyplot as plt

print(os.listdir("../input"))

# get data
train = pd.read_csv("../input/train.csv")
y = train['label']
X = train.drop(['label'], axis=1)

#X = X.values.reshape(-1,28,28)
X = X.values
y = y.values

# delete train to gain some space
del train

print("Shape of X:{0}".format(X.shape))
print("Shape of y:{0}".format(y.shape))
# get single digit graphical data
# to display it, we need to convert single line of 784 values to 28x28 square
digit_image = X[3].reshape(28,28)

plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
# split data for test and for training
# data before split_index will go for training and
#  data after split_index will go for testing
split_index = int(X.shape[0]*0.8)

X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

shuffle_index = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
y_train_5 = y_train == 5 # True for all 5s, False for all other digits
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42, max_iter=10)
sgd_clf.fit(X_train, y_train_5)
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

i=0;
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print("Correct ratio for fold {1}: {0}".format(n_correct / len(y_pred), i))
    i += 1

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()

cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

print("First 10 predictions of number 5: {0}".format(y_train_pred[:10]))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)

from sklearn.metrics import precision_score, recall_score
print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))
from sklearn.metrics import f1_score
score = f1_score(y_train_5, y_train_pred)
print(score)
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
print(y_scores)
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
y_scores = sgd_clf.decision_function([X_train[0]])
print("Score for 1st digit: {0}".format(y_scores[0]))
print("Was this digit a real 5? {0}".format(y_train_5[0]))

digit_image = X_train[0].reshape(28,28)
plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.title("Digit image")
plt.show()
threshold = -250000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
def print_recalls_precision(recalls, precisions, title):
    plt.figure(figsize=(8,6))
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.title("Precision vs Recall plot - {0}".format(title), fontsize=16)
    plt.axis([0,1,0,1])
    plt.show()
print_recalls_precision(recalls, precisions, "stochastic gradient descend")
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
# y_probas_forest contains 2 columns, one per class. Each row's sum of probabilities is equal to 1
y_scores_forest = y_probas_forest[:,1]

precisions_forest, recalls_forest, thresholds = precision_recall_curve(y_train_5, y_scores_forest)

print_recalls_precision(recalls_forest, precisions_forest, "random forest classifier")
never_5_predictions = cross_val_predict(never_5_clf, X_train, y_train_5, cv=3)
precisions_dumb, recalls_dumb, thresholds = precision_recall_curve(y_train_5, never_5_predictions)
print_recalls_precision(recalls_dumb, precisions_dumb, "dumb classifier")
plt.figure(figsize=(8,6))
plt.plot(precisions_forest, recalls_forest, "r-", label="Random Forest")
plt.plot(precisions, recalls, "g-", label="SGD classifier")
plt.plot(recalls_dumb, precisions_dumb, "b-", label="Dumb classifier")
plt.plot([0, 1], [1,0], "k--", label="Random guess")
plt.xlabel("Recall", fontsize=16)
plt.ylabel("Precision", fontsize=16)
plt.title("Precision vs Recall - model comparison", fontsize=16)
plt.axis([0,1,0,1])
plt.legend(loc="center left")
plt.ylim([0, 1])
print("F1 score for dumb classifier: {0}".format(f1_score(y_train_5, never_5_predictions)))
print("F1 score for SGD classifier: {0}".format(f1_score(y_train_5, y_train_pred)))
print("F1 score for Random Forest: {0}".format(f1_score(y_train_5, y_scores_forest > 0.5)))

