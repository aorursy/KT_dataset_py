# import required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

# import logistic regression model and accuracy_score metric

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score

from imblearn.over_sampling import SMOTE, SVMSMOTE
# Helper functions to compute and print metrics for classifier

def confusion_mat(y_true,y_pred, label='Confusion Matrix - Training Dataset'):

    print(label)

    cm = pd.crosstab(y_true, y_pred, rownames = ['True'],

                  colnames = ['Predicted'], margins = True)

    print(pd.crosstab(y_true, y_pred, rownames = ['True'],

                  colnames = ['Predicted'], margins = True))

    return cm



def metrics_clf(y_pred,y_true, print_metrics=True):

    acc=accuracy_score(y_true, y_pred)

    bal_acc=balanced_accuracy_score(y_true, y_pred)

    f1 =f1_score(y_true, y_pred)

    kappa = cohen_kappa_score(y_true, y_pred)

    if print_metrics:

        print(f'Accuracy score = {acc:.3f}\n')

        print(f'Balanced Accuracy score = {bal_acc:.3f}\n')

        print(f'F1 Accuracy score = {f1:.3f}\n')

        print(f'Cohen Kappa score = {kappa:.3f}\n')

    return (acc,bal_acc,f1, kappa)
# Show full output in cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity='all'
# Load data

data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
# Show five sampled records

data.sample(5)
# Show proportion of Classes

# 1 means Fraud, 0 Normal

_= data['Class'].value_counts().plot.bar(color=['coral', 'deepskyblue'])

data['Class'].value_counts()

print('Proportion of the classes in the data:\n')

print(data['Class'].value_counts() / len(data))
# Remove Time from data

data = data.drop(['Time'], axis = 1)

# create X and y array for model split

X = np.array(data[data.columns.difference(['Class'])])

y = np.array(data['Class']).reshape(-1, 1)

X

y

# split into training and testing datasets using stratify, i.e. same proportion class labels (0/1) in training and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 2, shuffle = True, stratify = y)
print('Proportion of the classes in training data:\n')

unique, counts = np.unique(y_train, return_counts=True)

print(f'"{unique[0]}": {counts[0]/len(y_train):.3f}')

print(f'"{unique[1]}": {counts[1]/len(y_train):.3f}')
print('Proportion of the classes in test data:\n')

unique, counts = np.unique(y_test, return_counts=True)

print(f'"{unique[0]}": {counts[0]/len(y_test):.3f}')

print(f'"{unique[1]}": {counts[1]/len(y_test):.3f}')
# standardize the data

# fit only on training data (to avoid data leakage)

scaler = StandardScaler()

scaler.fit(X_train)

X_train=scaler.transform(X_train)

X_test=scaler.transform(X_test)
# Fit a simple Logistic regression model

model_LR = LogisticRegression(solver = 'lbfgs')
# fit the model

model_LR.fit(X_train, y_train.ravel())



# prediction for training dataset

train_pred = model_LR .predict(X_train)



# prediction for testing dataset

test_pred = model_LR.predict(X_test)
(acc_train, b_acc_train, f1_train, k_train)=metrics_clf(y_train,train_pred)

cm_train=confusion_mat(y_train.ravel(),train_pred,'Confusion Matrix - Train Dataset (NO SMOTE)')
(acc_test_sm, b_acc_test_sm, f1_test_sm, k_test_sm)=metrics_clf(y_test,test_pred)

cm_test=confusion_mat(y_test.ravel(),test_pred,'Confusion Matrix - Test Dataset (NO SMOTE)')
sm = SMOTE(random_state = 42, n_jobs=-1, sampling_strategy='minority')

#sm= SVMSMOTE(random_state=42, k_neighbors=20, n_jobs=-1)
# generate balanced training data

# test data is left untouched

X_train_new, y_train_new = sm.fit_sample(X_train, y_train.ravel())



# observe that data has been balanced

ax = pd.Series(y_train_new).value_counts().plot.bar(title='Class distribution', y='Count',color=['coral', 'deepskyblue'])

_= ax.set_ylabel('Count')
# fit the model on balanced training data

_= model_LR.fit(X_train_new, y_train_new)



# prediction for Training data

train_pred_sm = model_LR.predict(X_train_new)



# prediction for Testing data

test_pred_sm = model_LR.predict(X_test, )
(acc_test_sm, b_acc_test_sm, f1_test_sm, k_test_sm)=metrics_clf(y_train_new,train_pred_sm)

cm_test=confusion_mat(y_train_new.ravel(),train_pred_sm,'Confusion Matrix - Train Dataset (SMOTE)')
(acc_test_sm, b_acc_test_sm, f1_test_sm, k_test_sm)=metrics_clf(y_test,test_pred_sm)

cm_test_sm=confusion_mat(y_test.ravel(),test_pred_sm,'Confusion Matrix - Test Dataset')