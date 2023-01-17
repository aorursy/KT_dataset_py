#My first public kernel learning about scikit-learn classification using the MNIST dataset - following tutorial 'Hands-on Machine Learning with Scikit-Learn & Tensorflow' by Aurelien Geron

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
mnist_train = pd.read_csv('../input/train.csv')
mnist_test = pd.read_csv('../input/test.csv')
print(mnist_train.info())
print(mnist_train.head())
X_train = np.array(mnist_train)[:, 1:785]
y_train = np.array(mnist_train)[:, 0]
print(X_train.shape)
print(y_train.shape)

X_test = np.array(mnist_test)[:, :]
print(X_test.shape)
#Visualise a digit
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

digit = X_train[37000]
digit_image = digit.reshape(28, 28)

plt.imshow(digit_image, cmap= matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
#Shuffle training set
shuffle_index = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
#Binary classifier Example
y_train_9 = (y_train == 9)
#y_test_9 = (y_test == 9)

#Stocastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_9)
sgd_clf.predict([digit])
#Cross-validation score
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_9, cv=3, scoring="accuracy") #Better than 90% accuracy
#Better way to evaluate preformance is the confusion matrix
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_9, cv=3)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_9, y_train_pred)

#35870 true negatives
#1942 false positives
#733 false negatives
#3455 true positives
#precison and recall metrics -> F1 score
from sklearn.metrics import precision_score, recall_score
pre = precision_score(y_train_9, y_train_pred)
rec = recall_score(y_train_9, y_train_pred)

print(pre, rec)

from sklearn.metrics import f1_score
f1 = f1_score(y_train_9, y_train_pred)

print(f1)
#Decision Function optimise SGDClassifier threshold
y_scores = cross_val_predict(sgd_clf, X_train, y_train_9, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_9, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])
plt.show()
y_train_pred_high = (y_scores > 400000)
precision_score(y_train_9, y_train_pred_high)
recall_score(y_train_9, y_train_pred_high)
#ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_train_9, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_train_9, y_scores))
#Test Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_9, cv=3, method="predict_proba")

y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_9, y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

print(roc_auc_score(y_train_9, y_scores_forest))
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_9, cv=3)

print(precision_score(y_train_9, y_train_pred_forest))
print(recall_score(y_train_9, y_train_pred_forest))
#Multiclass classifier
#SGDClassifiers
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([digit])
digit_scores = sgd_clf.decision_function([digit])
print(digit_scores)
sgd_clf.classes_
#Random Forests
forest_clf.fit(X_train, y_train)
print(forest_clf.predict([digit]))
print(forest_clf.predict_proba([digit]))
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
cross_val_score(forest_clf, X_train, y_train, cv=3, scoring="accuracy")
#Scaling input
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))
print(cross_val_score(forest_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))
#Here include other classifier algos and then hyperparameter optimisation
#Error Analysis
y_train_pred = cross_val_predict(forest_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
#Apply Random Forest Classifier on Test Set
#X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))
forest_clf.fit(X_train, y_train)
print(cross_val_score(forest_clf, X_train, y_train, cv=3, scoring="accuracy"))
output_pred = forest_clf.predict(X_test)
output = np.zeros((len(output_pred), 2))
for i in range(0, len(output)):
    #print(i)
    output[i, 0] = i+1
    output[i, 1] = output_pred[i]
    
print(output)
print(len(output))
output_pd = pd.DataFrame(data=output)
print(output_pd)
output_pd.to_csv('submission.csv', index = False)
#np.savetxt('../input/submission.txt', output, fmt='%i', delimiter=",", header="ImageID,Label")
'''
Current Submission with a simple multiclass Random Forest Classifier -> achieves 0.93828 ranked 2368/2718 (10/07/18)

Need to try other algorithms and hyperparameter optimisation, explore further uses of keras etc
'''