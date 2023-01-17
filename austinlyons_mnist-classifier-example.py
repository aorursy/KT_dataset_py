import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict

import sklearn.base

from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_curve, roc_auc_score

from sklearn.linear_model import SGDClassifier

from sklearn.datasets import fetch_openml

from sklearn.ensemble import RandomForestClassifier
#Gather data from the scikit-learn openml dataset and display the keys

mnist = fetch_openml('mnist_784', version = 1)

mnist.keys()
#Separate the data from the target labels into two different variables

X = mnist["data"]

y = mnist["target"]

print("Training data shape:", X.shape)

print("Target values shape:", y.shape)
#Examine the first row of training data in a 28x28 image. 

ex_train_val = X[0]

train_val_img = ex_train_val.reshape(28, 28)

plt.imshow(train_val_img, cmap = "binary")
#View the corresponding label for the first row of training data

y[0]
#Convert the labels into integers and separate the data into training and test sets. 

y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
#Let's focus on one digit (1) and attempt to classify a binary responses as either 1 or not 1. 

y_train_1 = (y_train == 1)

y_test_1 = (y_test == 1)
#Using SGD, attempt fit a model which will then correctly classify the training data as either a 1 or not 1.

sgd_cls = SGDClassifier(random_state = 1)

sgd_cls.fit(X_train, y_train_1)



#Now if we attempt to predict the first value in the training set we should return a false boolean since the first target is a 5.

sgd_cls.predict([ex_train_val])
#Using cross validation, evaluate the SGD Classifier on it ability to predict the nubmer 1. Visualize the  true and false negatives and positives in a confusion matrix

y_train_pred = cross_val_predict(sgd_cls, X_train, y_train_1, cv=3)

confusion_matrix(y_train_1, y_train_pred)
#Evaluate the models precision and recall using the F score

f1_score(y_train_1, y_train_pred)
y_scores = cross_val_predict(sgd_cls, X_train, y_train_1, cv=3, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_1, y_scores)
#Plot the precision and recall of the SGD model

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

    plt.grid(which="both",linestyle="--")

    plt.legend()



plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()
#Plot the ROC curve of the SGD Model

fpr, tpr, thresholds = roc_curve(y_train_1, y_scores)

def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0,1], [0,1], 'k--')

    plt.grid(which="both",linestyle="--")

    plt.legend()

    

plot_roc_curve(fpr, tpr)

plt.show()
roc_auc_score(y_train_1, y_scores)
#Compare the SGD Model to a Random Forest Model using ROC 

forest_cls = RandomForestClassifier(random_state=1)

y_prob_forest = cross_val_predict(forest_cls, X_train, y_train_1, cv=3, method="predict_proba")



y_scores_forest = y_prob_forest[:,1]

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_1, y_scores_forest)



plt.plot(fpr, tpr, "b:",label="SGD")

plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")

plt.legend(loc="lower right")

plt.show()
roc_auc_score(y_train_1, y_scores_forest)
#Train the Random Forest classifier and predict the outcomes of the test data

rnd_frst = RandomForestClassifier()

rnd_frst.fit(X_train, y_train)

output = rnd_frst.predict(X_test)
#Gather the results from the Random Forest Classifier and measure the accuracy of thee predictions

i = 0 

correct = 0

incorrect = 0

for i in range(len(output)):

    if output[i] == y_test[i]:

        correct += 1

    else:

        incorrect += 1

        

print("The accuracy of our model is:", correct / (correct + incorrect))