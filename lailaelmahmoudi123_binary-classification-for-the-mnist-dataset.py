# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

# Data Splitting Process

from sklearn.model_selection import train_test_split

# Training Process

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

# Performance Measures 

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
mnist_train = pd.read_csv("../input/digit-recognizer/train.csv")
mnist_test  = pd.read_csv("../input/digit-recognizer/test.csv")

#Take copies of the master dataframes

train = mnist_train.copy()
test = mnist_test.copy()
train.shape
test.shape
train.head()
train.tail()
test.head()
test.tail()
train.describe()
print(train.keys())
print(test.keys())
train.isnull().any().any()
X, y = train.drop(labels = ["label"],axis = 1).to_numpy(), train["label"]
X.shape
X.shape
y.shape
some_digit = X[20]
some_digit_show = plt.imshow(X[20].reshape(28,28), cmap=mpl.cm.binary)
y[20]
y = y.astype(np.uint8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
y_train_8 = (y_train == 8)
y_test_8 = (y_test == 8)
sgd_clf = SGDClassifier(max_iter=1000,random_state = 42)
sgd_clf.fit(X_train, y_train_8)
sgd_clf.predict([some_digit])
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train_8)
rf_clf.predict([some_digit])
cv_score_sgd = cross_val_score(sgd_clf, X_train, y_train_8, cv = 3, scoring = "accuracy")
cv_score_sgd = np.mean(cv_score_sgd)
cv_score_sgd
cv_score_rf = cross_val_score(rf_clf, X_train, y_train_8, cv= 3, scoring = "accuracy")
cv_score_rf = np.mean(cv_score_rf)
cv_score_rf
class Never8Classifier(BaseEstimator):
    def fit(sef, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_8_clf = Never8Classifier()

cross_val_score(never_8_clf, X_train, y_train_8, cv=3, scoring="accuracy")
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_8, cv= 3)


confusion_matrix(y_train_8, y_train_pred)
precision_score(y_train_8, y_train_pred)
recall_score(y_train_8, y_train_pred)
Score = f1_score(y_train_8, y_train_pred)
print(Score)
y_scores= cross_val_predict(sgd_clf, X_train, y_train_8, cv=3, method="decision_function")
print(y_scores)
precisions, recalls, thresholds = precision_recall_curve(y_train_8,y_scores)

# here we use matplotlib to plot recall and precision as functions of the thresholds

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    plt.title('Precision and recall versus the decision threshold')

    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
y_scores = sgd_clf.decision_function([X_train[0]])
print("Score for 1st digit: {0}".format(y_scores[0]))
print("Was this digit a real 8? {0}".format(y_train_8[0]))

digit_image = X_train[0].reshape(28,28)
plt.imshow(digit_image, cmap= matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.title("Digit image")
plt.show()
threshold = -200000
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
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(rf_clf, X_train, y_train_8, cv= 3, method= "predict_proba")
y_scores_forest = y_probas_forest[:,1]

# y_probas_forest contains 2 columns, one per class. Each row's sum of probabilities is equal to 1

precisions_forest, recalls_forest, thresholds = precision_recall_curve(y_train_8,y_scores_forest)
print_recalls_precision(recalls_forest, precisions_forest, "Random Forest Classifier")
never_8_predictions = cross_val_predict(never_8_clf, X_train, y_train_8, cv=3)

precisions_dumb, recalls_dumb, thresholds = precision_recall_curve(y_train_8, never_8_predictions)

print_recalls_precision(recalls_dumb, precisions_dumb, "dumb classifier")
plt.figure(figsize=(8,6))
plt.plot(precisions_forest, recalls_forest, "-r", label="Random Forest")
plt.plot(precisions,recalls, "-g",label="stochastic gradient descend")
plt.plot(precisions_dumb, recalls_dumb, "-b", label="dumb classifier")
plt.plot([0, 1], [1,0], "k--", label="Random guess")

plt.xlabel("Recall", fontsize=16)
plt.ylabel("precision", fontsize=16)


plt.title("Precision vs Recall - model comparison", fontsize=16)
plt.axis([0,1,0,1])
plt.legend(loc="center left")
plt.ylim([0, 1])
print("F1 score for dumb classifier: {0}".format(f1_score(y_train_8, never_8_predictions)))
print("F1 score for SGD classifier: {0}".format(f1_score(y_train_8, y_train_pred)))
print("F1 score for Random Forest: {0}".format(f1_score(y_train_8, y_scores_forest > 0.5)))
predictions_sgd = sgd_clf.predict(X_test).astype(int)
Label = pd.Series(predictions_sgd,name = 'Label')
ImageId = pd.Series(range(1,28001),name = 'ImageId')
submission = pd.concat([ImageId,Label],axis = 1)
submission.to_csv('submission.csv',index = False)
# clf = RandomForestClassifier(n_estimators=100, random_state=42)

# clf.fit(X_train, y_train_8)


predictions_forest = clf.predict(X_test).astype(int)

Label = pd.Series(predictions_forest,name = 'Label')
ImageId = pd.Series(range(1,28001),name = 'ImageId')
submission = pd.concat([ImageId,Label],axis = 1)
submission.to_csv('submission_forest.csv',index = False)