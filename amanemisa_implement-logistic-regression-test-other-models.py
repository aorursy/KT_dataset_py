import pandas as pd

import numpy as np
creditcard = pd.read_csv('../input/creditcard.csv')
X = creditcard[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',

       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]

y = creditcard['Class']
# we need train_test_split to split data into training set and test set

# we need metrics to measure accuracy after preditions

# we need optimize from scipy to optimize cost function

from sklearn.cross_validation import train_test_split

from sklearn import metrics

import scipy.optimize as op
# in order to save time, I keep the size of training set to be less than 100000

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.65, random_state=0)

print(Xtrain.shape);

print(Xtest.shape);
def sigmoid(z):

    return 1/(1 + np.exp(-z));
# define cost function. theta is an array containing coefficents for all feathers.

def costFunctionReg(theta, X, y):

    m = len(y)

    n = len(theta)

    h = sigmoid(X.dot(theta))

    J = (-y.T.dot(np.log(h))-(1-y.T).dot(np.log(1-h)))/m

    return J
# define gradient

def Gradient(theta, X, y):

    m = len(y)

    n = len(theta)

    h = sigmoid(X.dot(theta))

    grad = (1/m)*(X.T).dot(h-y);

    return grad.flatten()
# define predict function

def predict(theta, X):

    m, n = X.shape

    p = np.zeros(m)

    h = sigmoid(X.dot(theta))

    for i in range(0, m):

        if h[i] > 0.5:

            p[i] = 1

        else:

            p[i] = 0

    return p
# convert data to arrays

Xtrain = np.array(Xtrain)

ytrain = np.array(ytrain)

Xtest = np.array(Xtest)

ytest = np.array(ytest)
# add a column of ones to Xtrain

Xtrain_ones = np.append(np.ones((Xtrain.shape[0],1)), Xtrain, axis = 1)
# use fmin_bfgs to minimize cost function and fine theta, about 2 mins

initial_theta = np.zeros(Xtrain_ones.shape[1])

theta_optimal = op.fmin_bfgs(f= costFunctionReg, x0 = initial_theta, args = (Xtrain_ones,ytrain), fprime = Gradient, maxiter = 400);
# make predition and check accuracy

Xtest_ones = np.append(np.ones((Xtest.shape[0],1)), Xtest,axis = 1);

ypred = predict(theta_optimal,Xtest_ones);

print(metrics.confusion_matrix(ytest,ypred));

print(metrics.classification_report(ytest,ypred));

print('Accuracy : %f' %(metrics.accuracy_score(ytest,ypred)));

print('Area under the curve : %f' %(metrics.roc_auc_score(ytest,ypred)));
# call the classifier and train the data

from sklearn.linear_model import LogisticRegression

clf_logistic = LogisticRegression(penalty='l2');

clf_logistic.fit(Xtrain, ytrain);
# make predition and check accuracy

ypred = clf_logistic.predict(Xtest);

print(metrics.confusion_matrix(ytest,ypred));

print(metrics.classification_report(ytest,ypred));

print('Accuracy : %f' %(metrics.accuracy_score(ytest,ypred)));

print('Area under the curve : %f' %(metrics.roc_auc_score(ytest,ypred)));
from sklearn.svm import SVC
# SVC with 'linar' kernel. It took about 10 mins.

clf_linear = SVC(kernel='linear')

clf_linear.fit(Xtrain, ytrain)
# make prediction and check accuracy

ypred = clf_linear.predict(Xtest)

print(metrics.confusion_matrix(ytest,ypred))

print(metrics.classification_report(ytest,ypred))

print('Accuracy : %f' %(metrics.accuracy_score(ytest,ypred)))

print('Area under the curve : %f' %(metrics.roc_auc_score(ytest,ypred)))
# SVC with 'sigmoid' kernel

clf_sigmoid = SVC(kernel='sigmoid')

clf_sigmoid.fit(Xtrain, ytrain)
ypred = clf_sigmoid.predict(Xtest)

print(metrics.confusion_matrix(ytest,ypred));

print(metrics.classification_report(ytest,ypred));

print('Accuracy : %f' %(metrics.accuracy_score(ytest,ypred)));

print('Area under the curve : %f' %(metrics.roc_auc_score(ytest,ypred)));
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier()

clf_rf.fit(Xtrain,ytrain)
ypred = clf_rf.predict(Xtest);

print(metrics.confusion_matrix(ytest,ypred));

print(metrics.classification_report(ytest,ypred));

print('Accuracy : %f' %(metrics.accuracy_score(ytest,ypred)));

print('Area under the curve : %f' %(metrics.roc_auc_score(ytest,ypred)));
Xtrain2, Xtest2, ytrain2, ytest2 = train_test_split(X, y, test_size=0.2, random_state=0)

print(Xtrain2.shape);

print(Xtest2.shape);
# Use logistic regression again

clf_logistic2 = LogisticRegression(penalty='l2');

clf_logistic2.fit(Xtrain2, ytrain2);
# make predition and check accuracy

ypred2 = clf_logistic2.predict(Xtest2);

print(metrics.confusion_matrix(ytest2,ypred2));

print(metrics.classification_report(ytest2,ypred2));

print('Accuracy : %f' %(metrics.accuracy_score(ytest2,ypred2)));

print('Area under the curve : %f' %(metrics.roc_auc_score(ytest2,ypred2)));
# Use random forest classifier again

clf_rf2 = RandomForestClassifier()

clf_rf2.fit(Xtrain2,ytrain2);
ypred2 = clf_rf.predict(Xtest2);

print(metrics.confusion_matrix(ytest2,ypred2));

print(metrics.classification_report(ytest2,ypred2));

print('Accuracy : %f' %(metrics.accuracy_score(ytest2,ypred2)));

print('Area under the curve : %f' %(metrics.roc_auc_score(ytest2,ypred2)));