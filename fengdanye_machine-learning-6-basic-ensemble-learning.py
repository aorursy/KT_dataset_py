import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import os
print(os.listdir("../input"))
plt.rc('axes', lw = 1.5)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('xtick.major', size = 5, width = 3)
plt.rc('ytick.major', size = 5, width = 3)
wineData = pd.read_csv('../input/winequality-red.csv')

wineData['category'] = wineData['quality'] >= 7

X = wineData[wineData.columns[0:11]].values
y = wineData['category'].values.astype(np.int)
wineData.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)
# Below, random_state is only used to guarantee repeatable result for the tutorial. 
rfClf = RandomForestClassifier(n_estimators=500, random_state=0) # 500 trees. 
svmClf = SVC(probability=True, random_state=0) # force a probability calculation
logClf = LogisticRegression(random_state=0)

clf = VotingClassifier(estimators = [('rf',rfClf), ('svm',svmClf), ('log', logClf)], voting='soft') # construct the ensemble classifier
clf.fit(X_train, y_train) # train the ensemble classifier
from sklearn.metrics import precision_score, accuracy_score
y_true, y_pred = y_test, clf.predict(X_test)
print('precision on the test set: ', precision_score(y_true, y_pred))
print('accuracy on the test set: ', accuracy_score(y_true, y_pred))
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
phat = clf.predict_proba(X_test)[:,1]
plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, phat)
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
print('AUC is: ', auc(fpr,tpr))
wineData = pd.read_csv('../input/winequality-red.csv')

wineData['category'] = wineData['quality'] >= 7

X = wineData[wineData.columns[0:11]].values
y = wineData['category'].values.astype(np.int)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)
scaler = StandardScaler()
X_train_stan = scaler.fit_transform(X_train)
logReg = LogisticRegression(random_state=0, solver='lbfgs') # random_state is only set to guarantee for repeatable result for the tutorial
logReg.fit(X_train_stan, y_train)

X_test_stan = scaler.transform(X_test) # don't forget this step!
y_pred = logReg.predict(X_test_stan)

print('precision on the test set: ', precision_score(y_test, y_pred))
print('accuracy on the test set: ', accuracy_score(y_test, y_pred))
phat = logReg.predict_proba(X_test_stan)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, phat)

plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
print('AUC is: ', auc(fpr,tpr))
bagClf = BaggingClassifier(LogisticRegression(random_state=0, solver='lbfgs'), n_estimators = 500, oob_score = True, random_state = 90)
# again, random_state is only set to guarantee repeatable result for this tutorial.
bagClf.fit(X_train_stan, y_train)
print(bagClf.oob_score_) # The oob score is an estimate of the accuracy of the ensemble classifier, as introduced earlier. 
y_pred = bagClf.predict(X_test_stan)
phat = bagClf.predict_proba(X_test_stan)[:,1]

print('precision on the test set: ', precision_score(y_test, y_pred))
print('accuracy on the test set: ', accuracy_score(y_test, y_pred))
fpr, tpr, thresholds = roc_curve(y_test, phat)
plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
print('AUC is: ', auc(fpr,tpr))
wineData.head(0)
bagClf = BaggingClassifier(LogisticRegression(random_state=0, solver='lbfgs'), n_estimators = 500, 
                           bootstrap_features = True, max_features = 1.0, oob_score = True, random_state = 90)
# Notice that bootstrap_features is set to True.
bagClf.fit(X_train_stan, y_train)
print(bagClf.oob_score_) # The oob score is an estimate of the accuracy of the ensemble classifier
y_pred = bagClf.predict(X_test_stan)
phat = bagClf.predict_proba(X_test_stan)[:,1]

print('precision on the test set: ', precision_score(y_test, y_pred))
print('accuracy on the test set: ', accuracy_score(y_test, y_pred))
fpr, tpr, thresholds = roc_curve(y_test, phat)
plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
print('AUC is: ', auc(fpr,tpr))
tuned_parameters = {'n_estimators':[500],'n_jobs':[-1], 'max_features': [0.5,0.6,0.7,0.8,0.9,1.0], 
                    'max_depth': [10,11,12,13,14],'min_samples_leaf':[1,10,100],'random_state':[0]} 

clf = GridSearchCV(ExtraTreesClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
clf.fit(X_train, y_train)
print('The best model is: ', clf.best_params_)
print('This model produces a mean cross-validated score (auc) of', clf.best_score_)
y_pred = clf.predict(X_test)
print('precision on the evaluation set: ', precision_score(y_test, y_pred))
print('accuracy on the evaluation set: ', accuracy_score(y_test, y_pred))
phat = clf.predict_proba(X_test)[:,1]
plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, phat)
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
print('AUC is: ', auc(fpr,tpr))
wineData = pd.read_csv('../input/winequality-red.csv')
wineData.head()
X = wineData[wineData.columns[0:11]].values
y = wineData['quality'].values.astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)

print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)
# Standardize the data
scaler = StandardScaler()
X_train_stan = scaler.fit_transform(X_train)
X_test_stan = scaler.transform(X_test)
# use an ensemble of 500 linear regressors. Use Random Patches method.
bagReg = BaggingRegressor(LinearRegression(), n_estimators = 500, 
                           bootstrap_features = True, max_features = 1.0, oob_score = True, random_state = 0)

bagReg.fit(X_train_stan, y_train)
print("oob score is: ", bagReg.oob_score_) # The oob score is an estimate of the r2 score of the ensemble classifier
from sklearn.metrics import r2_score
y_pred = bagReg.predict(X_test_stan)
print("The r2 score on the test set is: ", r2_score(y_test, y_pred))
tuned_parameters = {'n_estimators':[500],'n_jobs':[-1], 'max_features': [0.5,0.6,0.7,0.8,0.9,1.0], 
                    'max_depth': [16,20,24],'min_samples_leaf':[1,10,100],'random_state':[0]} 

reg = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring='r2')
reg.fit(X_train, y_train)
print('The best model is: ', reg.best_params_)
print('This model produces a mean cross-validated score (r2) of', reg.best_score_)
y_pred = reg.predict(X_test)
print("The r2 score on the test set is: ", r2_score(y_test, y_pred))
tuned_parameters = {'n_estimators':[500],'n_jobs':[-1], 'max_features': [0.5,0.6,0.7,0.8,0.9,1.0], 
                    'max_depth': [20,24,28],'min_samples_leaf':[1,10,100],'random_state':[0]} 

reg = GridSearchCV(ExtraTreesRegressor(), tuned_parameters, cv=5, scoring='r2')
reg.fit(X_train, y_train)
print('The best model is: ', reg.best_params_)
print('This model produces a mean cross-validated score (r2) of', reg.best_score_)
y_pred = reg.predict(X_test)
print("The r2 score on the test set is: ", r2_score(y_test, y_pred))
plt.plot(y_test,y_pred, linestyle='',marker='o')
plt.xlabel('true y values', fontsize = 14)
plt.ylabel('predicited y values', fontsize = 14)
plt.show()