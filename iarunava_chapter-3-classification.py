import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import os
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return x_train, y_train, x_test, y_test

X_train, y_train, X_test, y_test = load_data('../input/mnist.npz')

num_train = X_train.shape[0]
num_test = X_test.shape[0]
num_labels = 10

print ('X_train shape', X_train.shape)
print ('y_train shape', y_train.shape)
print ('X_test shape', X_test.shape)
print ('y_test shape', y_test.shape)
random_num = 10
plt.figure(figsize=(30, 30))
for i in range(random_num):
    rand_index = np.random.choice(num_train, replace=False)
    plt.subplot(random_num, np.ceil(random_num/2), i+1)
    plt.imshow(X_train[rand_index])
    plt.axis('off')
    plt.title(y_train[rand_index])
plt.show()
X_train = X_train.reshape(X_train.shape[0], -1)
y_train = y_train.reshape(y_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
y_test = y_test.reshape(y_test.shape[0], -1)

print ('X_train shape', X_train.shape)
print ('y_train shape', y_train.shape)
print ('X_test shape', X_test.shape)
print ('y_test shape', y_test.shape)
def shuffle_data(X, y):
    assert X.shape[0] == y.shape[0]
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    return X, y

X_train, y_train = shuffle_data(X_train, y_train)
X_test, y_test = shuffle_data(X_test, y_test)
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train_5)
rand_index = np.random.choice(num_test)
pred = sgd_clf.predict(X_test[rand_index].reshape(1, -1))

plt.imshow(X_test[rand_index].reshape(28, 28))
plt.axis('off')
plt.title(y_test_5[rand_index])
plt.show()

print ('Prediction wassss', pred == y_test_5[rand_index])
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    
    def predict(self, X):
        return np.zeros(X.shape[0], dtype=bool)
never_5_clf = Never5Classifier()
never_5_clf.fit(X_train, y_train)
cross_val_score(never_5_clf, X_train, y_train_5, cv=5, scoring='accuracy')
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)
confusion_matrix(y_train_5, y_train_5)
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
f1_score(y_train_5, y_train_pred)
rand_index = 439#np.random.choice(num_train)

y_scores = sgd_clf.decision_function(X_train[rand_index].reshape(1, -1))
print ('y_scores: ', y_scores)
threshold = 0
print ('Image is of a 5: ', y_scores > threshold)
y_train_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
print (y_train_scores.shape)
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_train_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend()
    plt.ylim([-0.1, 1.1])
    return

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
def precision_vs_recall_curve(precisions, recalls):
    plt.plot(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision/Recall Tradeoff   |   PR Curve')
    return

precision_vs_recall_curve(precisions, recalls)
plt.show()
y_train_pred_90_precision = y_train_scores > 70000
precision_score(y_train_5, y_train_pred_90_precision)
recall_score(y_train_5, y_train_pred_90_precision)
fpr, tpr, thresholds = roc_curve(y_train_5, y_train_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=None)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    return

plot_roc_curve(fpr, tpr)
plt.show()
roc_auc_score(y_train_5, y_train_scores)
forest_clf = RandomForestClassifier()
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')
print (y_probas_forest.shape)
y_scores_forest = y_probas_forest[:, 1] #score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
plt.plot(fpr, tpr, 'b:', label='SGD')
plt.plot(fpr_forest, tpr_forest, label='Random Forest')
plt.plot([0, 1], [0, 1], '--')
plt.legend(loc='bottom right')
plt.show()
roc_auc_score(y_train_5, y_scores_forest)
precision_score(y_train_5, np.argmax(y_probas_forest, axis=1))
recall_score(y_train_5, np.argmax(y_probas_forest, axis=1))
rand_index = X_train[np.random.choice(X_train.shape[0])]

sgd_clf.fit(X_train, y_train)
print ('Predicted number is: ', sgd_clf.predict(rand_index.reshape(1, -1)))

plt.imshow(rand_index.reshape(28, 28))
plt.axis('off')
plt.show()
rand_index = X_train[np.random.choice(X_train.shape[0])]

rand_index_scores = sgd_clf.decision_function(rand_index.reshape(1, -1))
print (rand_index_scores)

plt.imshow(rand_index.reshape(28, 28))
plt.axis('off')
plt.show()
sgd_clf.classes_
ovo_clf = OneVsOneClassifier(SGDClassifier())
ovo_clf.fit(X_train, y_train)
ovo_clf.predict(rand_index.reshape(1, -1)) #rand_index is from the above example
len(ovo_clf.estimators_)
forest_clf.fit(X_train, y_train)
forest_clf.predict(rand_index.reshape(1, -1))
forest_clf.predict_proba(rand_index.reshape(1, -1))
cross_val_score(sgd_clf, X_train, y_train, cv=5, scoring='accuracy')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy')
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print (conf_mx)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
print (norm_conf_mx)
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
cl_a, cl_b = 3, 5

X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8, 8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
rand_index = X_train[np.random.choice(X_train.shape[0])]

plt.imshow(rand_index.reshape(28, 28))
plt.axis('off')
plt.show()

knn_clf.predict(rand_index.reshape(1, -1))
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
f1_score(y_train, y_train_knn_pred, average='macro')
noise_train = np.random.randint(0, 100, (X_train.shape[0], 784))
noise_test = np.random.randint(0, 100, (X_test.shape[0], 784))

X_train_noise = X_train + noise_train
y_train_noise = X_train
X_test_noise = X_test + noise_test
y_test_noise = X_test
rand_index = np.random.choice(X_train_noise.shape[0])

plt.subplot(121)
plt.imshow(X_train_noise[rand_index].reshape(28, 28))
plt.axis('off')
plt.subplot(122)
plt.imshow(X_train[rand_index].reshape(28, 28))
plt.axis('off')
plt.show()
knn_clf.fit(X_train_noise, y_train_noise)

rand_index = np.random.choice(X_test_noise.shape[0])
clean_digit = knn_clf.predict(X_test_noise[rand_index].reshape(1, -1))

plt.figure(figsize=(10, 10))

plt.subplot(131)
plt.imshow(X_test_noise[rand_index].reshape(28, 28))
plt.title('Noisy Digit')
plt.axis('off')

plt.subplot(132)
plt.imshow(y_test_noise[rand_index].reshape(28, 28))
plt.title('Clean target digit')
plt.axis('off')

plt.subplot(133)
plt.imshow(clean_digit.reshape(28, 28))
plt.title('Predicted Clean Digit')
plt.axis('off')

plt.show()