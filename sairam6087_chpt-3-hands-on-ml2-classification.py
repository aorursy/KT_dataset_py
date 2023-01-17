import pandas as pd

test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
X_train, y_train = train.drop(labels=["label"],axis=1),  train["label"]

X_test, y_test = test.drop(labels=["label"],axis=1),  test["label"]
X_train = X_train.values.reshape(-1, 784)

X_test = X_test.values.reshape(-1, 784)
import numpy as np

import sklearn

import os

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt



np.random.seed(42)
some_digit = X_train[0]



def plot_digit(data):

    image = data.reshape(28,28)

    plt.imshow(image, cmap="binary")

    plt.axis("off")

    plt.show()

    

plot_digit(some_digit)
def plot_digits(instances, images_per_row=10, **options):

    size = 28

    images_per_row = min(len(instances), images_per_row)

    images = [instance.reshape(size,size) for instance in instances]

    n_rows = (len(instances) - 1) // images_per_row + 1

    row_images = []

    n_empty = n_rows * images_per_row - len(instances)

    images.append(np.zeros((size, size * n_empty)))

    for row in range(n_rows):

        rimages = images[row * images_per_row : (row + 1) * images_per_row]

        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)

    plt.imshow(image, cmap = mpl.cm.binary, **options)

    plt.axis("off")
plt.figure(figsize=(10,10))

example_images = X_train[:100]

plot_digits(example_images, images_per_row=10)

plt.show()
# Setup Training Set

y_train_5 = (y_train == 5)

y_test_5 = (y_test == 5)
from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train_5)
# Test on our example above

sgd_clf.predict([some_digit])
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
from sklearn.model_selection import StratifiedKFold

from sklearn.base import clone



skfolds = StratifiedKFold(n_splits=3, random_state=42)



for train_index, test_index in skfolds.split(X_train, y_train_5):

    clone_clf = clone(sgd_clf) # Clone the model for each fold's run

    X_train_fold = X_train[train_index]

    y_train_fold= y_train_5[train_index]

    X_test_fold = X_train[test_index]

    y_test_fold = y_train_5[test_index]

    

    clone_clf.fit(X_train_fold, y_train_fold)

    y_pred = clone_clf.predict(X_test_fold)

    

    num_correct = sum(y_test_fold == y_pred)

    print(num_correct/len(y_test_fold))
from sklearn.base import BaseEstimator



class Never5Classifier(BaseEstimator):

    def fit(self, X, y=None):

        pass

    def predict(self, X):

        return np.zeros((len(X),1), dtype=bool)

    
never_5_clf = Never5Classifier()

cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy",n_jobs=-1) # Clearly something's up here! Just guessing not 5 results in > 90% accuracy!!
y_train_5.value_counts() # There's only 10% True labels in the dataset => Accuracy isn't ideal for measuring performance
# For the confusion matrix, we need the actual predictions themselves

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,n_jobs=-1)



from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)
# In a perfect world, the confusion matrix of a perfect classifier would be

y_perfect = y_train_5

confusion_matrix(y_train_5, y_perfect)
from sklearn.metrics import precision_score, recall_score

# Precision

precision_score(y_train_5, y_train_pred) # == 3530 / (3530 + 687) ; Look at the confusion matrix result above [[TN FN], [FP TP]] for these numbers
3530/(3530 + 687)
# Recall

recall_score(y_train_5, y_train_pred) # == 3530 / (3530 + 1891)
3530/(3530 + 1891)
# F1 score is the harmonic mean of precision and recall 

from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)
3530 / (3530 + (687 + 1891)/2) # Verify mathematically
# Let's threshold the scores

y_score = sgd_clf.decision_function([some_digit])

y_score
# Clearly depending on the threshold set, the prediction changes

threshold_low, threshold_high = 0, 4000

pred_1, pred_2 = (y_score > threshold_low), (y_score > threshold_high)

print(pred_1, pred_2)
from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function",n_jobs=-1) # Get the raw scores instead of the labels

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
# Option A

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1],"b--", label="Precision", linewidth=2)

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall",linewidth=2)

    plt.legend(loc="center right", fontsize=16)

    plt.xlabel("Threshold", fontsize=16)

    plt.grid(True)

    plt.axis([-50000, 50000, 0, 1])
recall_90_precision = recalls[np.argmax(precisions >= 0.90)]

threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]





plt.figure(figsize=(8, 4))                                                                  

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")                 

plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")                                

plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")

plt.plot([threshold_90_precision], [0.9], "ro")                                            

plt.plot([threshold_90_precision], [recall_90_precision], "ro")                             

plt.title("precision_recall_vs_threshold_plot")                                              

plt.show()
# Option B

def plot_precision_vs_recall(precisions, recalls):

    plt.plot(recalls, precisions, "b-", linewidth=2)

    plt.xlabel("Recall", fontsize=16)

    plt.ylabel("Precision", fontsize=16)

    plt.axis([0, 1, 0, 1])

    plt.grid(True)



plt.figure(figsize=(8, 6))

plot_precision_vs_recall(precisions, recalls)

plt.plot([0.4368, 0.4368], [0., 0.9], "r:")

plt.plot([0.0, 0.4368], [0.9, 0.9], "r:")

plt.plot([0.4368], [0.9], "ro")

plt.title("precision_vs_recall_plot")

plt.show()
y_train_pred_90 = (y_scores >= threshold_90_precision)
precision_score(y_train_5, y_train_pred_90) # Good Precision
recall_score(y_train_5, y_train_pred_90) # But at the cost of poor recall
from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0,1], [0, 1], 'k--')

    plt.axis([0, 1, 0, 1])

    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=16)

    plt.ylabel("True Positive Rate (Recall)", fontsize=16)

    plt.grid(True)

    



plt.figure(figsize=(8,6))

plot_roc_curve(fpr, tpr)

plt.plot([4.837e-3, 4.837e-3], [0., 0.4368], "r:") 

plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:") 

plt.plot([4.837e-3], [0.4368], "ro")            

plt.title("ROC Curve")                         

plt.show()
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba",n_jobs=-1)
y_scores_forest = y_probas_forest[:, 1]

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest) # Clearly the Random Forest is better :)
plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")

plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")

plt.plot([4.837e-3, 4.837e-3], [0., 0.4368], "r:")

plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:")

plt.plot([4.837e-3], [0.4368], "ro")

plt.plot([4.837e-3, 4.837e-3], [0., 0.9487], "r:")

plt.plot([4.837e-3], [0.9487], "ro")

plt.grid(True)

plt.legend(loc="lower right", fontsize=16)

plt.title("ROC Curve Comparison")

plt.show()
# Let's see how this lines up with the precision and recall values

roc_auc_score(y_train_5, y_scores_forest)
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,n_jobs=-1)

precision_score(y_train_5, y_train_pred_forest) # Good Precision
recall_score(y_train_5, y_train_pred_forest) # At much better recall than the SGD
from sklearn.svm import SVC



svm_clf = SVC(gamma="auto", random_state=42)

svm_clf.fit(X_train[:1000], y_train[:1000]) # Training with the whole dataset takes forever! :D

svm_clf.predict([some_digit]) # What does it predict for "5"
# What are the scores for this sample?

some_digit_scores = svm_clf.decision_function([some_digit])

some_digit_scores
# What is the index with the max score?

np.argmax(some_digit_scores)
# What are the class labels for each index?

svm_clf.classes_
# What is in the 5th index?

svm_clf.classes_[5]
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC())

ovr_clf.fit(X_train[:1000], y_train[:1000])

ovr_clf.predict([some_digit])
# How many estimators do we have?

len(ovr_clf.estimators_)
# The same strategy of One vs One is used even for SGD by default

sgd_clf.fit(X_train[:5000], y_train[:5000])

sgd_clf.predict([some_digit])
sgd_clf.decision_function([some_digit])
# What are the scores?

cross_val_score(sgd_clf, X_train[:5000], y_train[:5000], cv=3, scoring="accuracy", n_jobs=-1)
# What if we used Scaling?

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

cross_val_score(sgd_clf, X_train_scaled[:5000], y_train[:5000], cv=3, scoring="accuracy", n_jobs=-1) # Scores improve :D
# Confusion matrix

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled[:5000], y_train[:5000], cv=3, n_jobs=-1)

conf_mx = confusion_matrix(y_train[:5000], y_train_pred[:5000])

conf_mx
# How does it look like?

plt.matshow(conf_mx, cmap=plt.cm.gray)

plt.show()
row_sums = conf_mx.sum(axis=1, keepdims=True)

norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)  # Let's keep only the errors

plt.matshow(norm_conf_mx, cmap=plt.cm.gray)

plt.show()
cl_a, cl_b = 3, 5

X_aa = X_train[:5000][(y_train[:5000] == cl_a) & (y_train_pred == cl_a)]

X_ab = X_train[:5000][(y_train[:5000] == cl_a) & (y_train_pred == cl_b)]

X_ba = X_train[:5000][(y_train[:5000] == cl_b) & (y_train_pred == cl_a)]

X_bb = X_train[:5000][(y_train[:5000] == cl_b) & (y_train_pred == cl_b)]



plt.figure(figsize=(8,8))

plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)

plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)

plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)

plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)

plt.show()
from sklearn.neighbors import KNeighborsClassifier



# Problem Statement: Is the digit larger than 6. Also is it odd?

y_train_large = (y_train >= 7)

y_train_odd = (y_train % 2 == 1)



y_multilabel = np.c_[y_train_large, y_train_odd]



knn_clf = KNeighborsClassifier()

knn_clf.fit(X_train[:5000], y_multilabel[:5000, :])

# What is 5 ?

knn_clf.predict([some_digit]) # It is not > 6 and it is odd
y_train_knn_pred = cross_val_predict(knn_clf, X_train[:5000], y_multilabel[:5000, :], cv=3, n_jobs=-1)

f1_score(y_multilabel[:5000, :], y_train_knn_pred, average="macro")
# Problem Statement : Image denoising



# Add noise to the input images

noise = np.random.randint(0, 100, (len(X_train), 784))

X_train_mod = X_train + noise

noise = np.random.randint(0, 100, (len(X_test), 784))

X_test_mod = X_test + noise

y_train_mod = X_train

y_test_mod = X_test
some_index = 108

plt.subplot(121); plot_digit(X_test_mod[some_index])

plt.subplot(122); plot_digit(y_test_mod[some_index])

plt.show()
knn_clf.fit(X_train_mod[:5000,:], y_train_mod[:5000,:])

clean_digit = knn_clf.predict([X_test_mod[some_index]])

plot_digit(clean_digit)

plt.show() # Not bad :D