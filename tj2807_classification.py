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
# sklearn provides an ability to download datasets using ftch_openml

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version =1)

mnist.keys()
X, y = mnist['data'], mnist['target']

print(f'shape of the Data: {X.shape}')

print(f'shape of Labels: {y.shape}')
import matplotlib as mpl

import matplotlib.pyplot as plt



nImage = 35

some_image = X[nImage].reshape(28, 28)

plt.imshow(some_image, cmap = mpl.cm.binary, interpolation = 'nearest')

plt.axis('off')

plt.show()

print(y[nImage])
y = y.astype(np.uint8)

y.dtype
# separate the test set. mnist dataset is already shuffled and train and test set is separated. First 60k images are training set. This will help us get uniform CV folds.

# some algorithms are also sensitive to order of training examples and perform poorly if too many same labels are in a row. This also can be avoided.



X_train, y_train, X_test, y_test = X[:60000], y[:60000], X[60000:], y[60000:]
# target vectors

y_train_5 = (y_train==5)

y_tst_5 = (y_test==5)
# Train a SGDClassifier



from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([X[nImage]])
len(X_train)
# implementing my own cross validation function for more control



from sklearn.model_selection import StratifiedKFold

from sklearn.base import clone



def do_cross_validation(data, target, classifier):

    skfolds = StratifiedKFold(n_splits=3, random_state = 42)

    accuracy = []

    for train_index, test_index in skfolds.split(data, target):

        clone_clf = clone(classifier)

        x_train_cv = X_train[train_index]

        y_train_cv = y_train_5[train_index]

        x_test_cv = X_train[test_index]

        y_test_cv = y_train_5[test_index]

        clone_clf.fit(x_train_cv, y_train_cv)

        predictions = clone_clf.predict(x_test_cv)

        n_correct = np.sum(predictions == y_test_cv)

        accuracy.append(n_correct/(len(predictions)))

    return accuracy



do_cross_validation(X_train, y_train_5, sgd_clf)
# Let's try out a just give not 5 classifier

from sklearn.base import BaseEstimator



class justGiveNotFive(BaseEstimator):

    def fit(self, X, y=None):

        pass

    def predict(self, X):

        return np.zeros((len(X)), dtype = bool)

    

jgnf_clf = justGiveNotFive()

do_cross_validation(X_train, y_train_5, jgnf_clf)



# from sklearn.model_selection import cross_val_score



# cross_val_score(jgnf_clf, X_train, y_train_5, cv=3, scoring='accuracy')
np.sum(y_train_5)/len(y_train_5)



# Just around 9% digits are 5.
# Confusion matrix gives all combinations of predicted vs actual classes.

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_predict



# cross_val_predict does cross validation folding, just instead of returning the evaluation scores in each fold,

# it returns the predictions.

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv= 4)



confusion_matrix(y_train_pred, y_train_5)
# Assume we reached perfection

y_train_perfect_predictions = y_train_5

confusion_matrix(y_train_5, y_train_perfect_predictions)



# We will see only True negatives on top left corner and true positives in bottom right.
# Precision and Recall

from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
# F1 score is the combination of Precision and Recall

from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)
# Checking the precision recall tradeoff by varying the threshold

# Calling estimator's decision function gives us the score it uses to make classification



y_scores = sgd_clf.decision_function(some_image.reshape(1, -1))

threshold = 0

y_some_digit_pred = (y_scores > threshold)

y_some_digit_pred
# if threshold is very high, this same digit will get predicted as false

# This represents lower recall

threshold = 8000

y_scores > threshold
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3, method='decision_function')
# How to decide which threshold to use?



from sklearn.metrics import precision_recall_curve



precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
def plot_precision_recall_vc_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')

    plt.plot(thresholds, recalls[:-1], 'g-', label="Recall")

    

plot_precision_recall_vc_threshold(precisions, recalls, thresholds)
# suppose we wish to choose a threshold such that precision is 90%



precision90_threshold = thresholds[np.argmax(precisions >= 0.90)]

y_pred90 = (y_scores >= precision90_threshold)

precision_score(y_train_5, y_pred90 )
recall_score(y_train_5, y_pred90)
# ROC curve

from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)



def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0,1], [0,1], 'k--')

    plt.grid(True)

    plt.ylabel('True Positive Rate (Recall)')

    plt.xlabel('False Positive Rate')

    

    

plot_roc_curve(fpr, tpr)

plt.show()
# Try random forest

from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method = 'predict_proba')



y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
plt.plot(fpr, tpr, "b:", label='SGD')

plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")

plt.legend(loc="lower right")

plt.show()
# Random forest is clearly performing better than SGD

from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores_forest)
X_train.shape
y_train.shape
# when using binary classifier for multiClass classification, sklearn automatically does OvA classification.



sgd_clf.fit(X_train, y_train)
sgd_clf.predict(some_image.reshape(1,-1))

some_digit_scores = sgd_clf.decision_function(some_image.reshape(1, -1))

some_digit_scores

# sklearn ran this image for all 10 classes and chose the one with highest score. which was 5.
# to use one vs one classfier, we use that class and pass it our binary classifier

from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))

ovo_clf.fit(X_train, y_train)

ovo_clf.predict(some_image.reshape(1,-1))
# classfier trained Nc2 models

len(ovo_clf.estimators_)
# Random Forest classfier can do multiclass classification directly

forest_clf.fit(X_train, y_train)

forest_clf.predict(some_image.reshape(1,-1))
forest_clf.predict_proba(some_image.reshape(1,-1))
# metric accuracy

cross_val_score(forest_clf, X_train, y_train, cv = 3, scoring = 'accuracy')
# Scaling the input may help

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

cross_val_score(forest_clf, X_train_scaled, y_train, cv = 3, scoring = 'accuracy')
y_train_pred = cross_val_predict(forest_clf, X_train_scaled, y_train, cv=3)

conf_mx = confusion_matrix(y_train, y_train_pred)

conf_mx
plt.matshow(conf_mx, cmap=plt.cm.gray)
# but this matrix shows absolute no. of errors, so it's unfair for classes which have large no. of examples.

# dividing each no with total values in class



row_sums = conf_mx.sum(axis = 1, keepdims = True)

norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0) # keep only errors

plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# sometimes we may want multi label outputs like 2 outputs - 

# whether digit is large, whether it is odd



from sklearn.neighbors import KNeighborsClassifier



y_train_large = (y_train >=7)

y_train_odd = (y_train % 2 == 1)

y_multi = np.c_[y_train_large, y_train_odd]



knn_clf = KNeighborsClassifier()

knn_clf.fit(X_train, y_multi)
knn_clf.predict(some_image.reshape(1,-1))
# measure F1 score

# Assumes that both labels equally important when averaging

# y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multi, cv=3)

# f1_score(y_multi, y_train_knn_pred, average='macro')
# suppose we take as an input a noisy image and output a clean image, there

# will be 784 labels with each label taking value between 0-255



noise = np.random.randint(0, 100, (len(X_train), 784))

X_train_mod = X_train + noise

noise = np.random.randin(0,100, (len(X_test), 784))

X_test_mod = X_test + noise

y_train_mod = X_train

y_test_mod = X_test
knn_clf.fit(X_train_mod, y_train_mod)

clean_digit = knn_clf.predict([X_test_mod[nImage]])

plot_digit(clean_digit)