# Importing some libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
# Let's make the example dataset

X,y = make_classification(n_samples = 1000, n_classes = 2, weights = [1,1], random_state = 1)
# Let's check shape of the dataset created

print(X.shape)    # ----> Independent variables

print(y.shape)    # ----> Dependent variable/Labels
# Let's split the data into train and test sets

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 56)
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)

y_train_pred = rf_model.predict_proba(X_train)

y_test_pred = rf_model.predict_proba(X_test)

print(f"Random Forest train roc_auc : {roc_auc_score(y_train, y_train_pred[:,1])}")

print(f"Random Forest test roc_auc : {roc_auc_score(y_test, y_test_pred[:,1])}")
y_train_pred
from sklearn.linear_model import LogisticRegression

log_classifier = LogisticRegression()

log_classifier.fit(X_train, y_train)

y_train_pred = log_classifier.predict_proba(X_train)

y_test_pred = log_classifier.predict_proba(X_test)

print(f"Logistic Regression train roc_auc : {roc_auc_score(y_train, y_train_pred[:,1])}")

print(f"Logistic Regression test roc_auc : {roc_auc_score(y_test, y_test_pred[:,1])}")
from sklearn.ensemble import AdaBoostClassifier

adb_classifier = AdaBoostClassifier()

adb_classifier.fit(X_train, y_train)

y_train_pred = adb_classifier.predict_proba(X_train)

y_test_pred = adb_classifier.predict_proba(X_test)

print(f"ADA Boost Classifier train roc_auc : {roc_auc_score(y_train, y_train_pred[:,1])}")

print(f"ADA Boost Classifier test roc_auc : {roc_auc_score(y_test, y_test_pred[:,1])}")
from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier()

knn_classifier.fit(X_train, y_train)

y_train_pred = knn_classifier.predict_proba(X_train)

y_test_pred = knn_classifier.predict_proba(X_test)

print(f"KNN Classifier train roc_auc : {roc_auc_score(y_train, y_train_pred[:,1])}")

print(f"KNN Classifier test roc_auc : {roc_auc_score(y_test, y_test_pred[:,1])}")
# Taking the mean of the predictions made by all the models

pred = []

for model in [rf_model,log_classifier,adb_classifier,knn_classifier]:

    pred.append(pd.Series(model.predict_proba(X_test)[:,1]))

final_prediction = pd.concat(pred,axis = 1).mean(axis = 1)

print(f"Ensemble test roc-auc: {roc_auc_score(y_test,final_prediction)}")
# Calculating the roc curve values

fpr,tpr,thresholds = roc_curve(y_test, final_prediction)
thresholds
# Checking the accuracy of the model with respect to each of the threshold values

# Here, we are assigning the values in the final_prediction to 1 if > threshold otherwise 0



from sklearn.metrics import accuracy_score

accuracy = []

for thres in thresholds:

    y_pred = np.where(final_prediction > thres,1,0)

    accuracy.append(accuracy_score(y_test, y_pred, normalize = True))

    

accuracy = pd.concat([pd.Series(thresholds), pd.Series(fpr), pd.Series(tpr), pd.Series(accuracy)],

                        axis = 1)

accuracy.columns = ['Thresholds', 'FPR', 'TPR', 'Accuracy']

accuracy.sort_values(by ='Accuracy', ascending = False, inplace = True)

accuracy.reset_index(drop = True,inplace = True)
accuracy
# Plot showing roc curve

def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
plot_roc_curve(fpr,tpr)