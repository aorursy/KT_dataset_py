import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_score, accuracy_score, auc, roc_curve, r2_score



import os

print(os.listdir("../input"))
plt.rc('axes', lw = 1.5)

plt.rc('xtick', labelsize = 14)

plt.rc('ytick', labelsize = 14)

plt.rc('xtick.major', size = 5, width = 3)

plt.rc('ytick.major', size = 5, width = 3)
data = pd.read_csv('../input/winequality-red.csv')

data['category'] = data['quality'] >= 7 # again, binarize for classification

data.head()
X = data[data.columns[0:11]].values

y = data['category'].values.astype(np.int)



# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42) 

# random_state is fixed to guarantee repeatable results for this tutorial - remove it in actual practice



print('X train size: ', X_train.shape)

print('X test size: ', X_test.shape)

print('y train size: ', y_train.shape)

print('y test size: ', y_test.shape)
adaClf = AdaBoostClassifier(algorithm = 'SAMME', random_state=12)

adaClf.fit(X_train,y_train)
p_pred = adaClf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, p_pred)



plt.subplots(figsize=(8,6))

plt.plot(fpr, tpr)

plt.xlim(0,1)

plt.ylim(0,1)

plt.xlabel('False positive rate', fontsize = 16)

plt.ylabel('True positive rate', fontsize = 16)

plt.show()
print('AUC is: ', auc(fpr, tpr))
print(adaClf.estimator_weights_)
print(adaClf.estimator_errors_)
for estimator in adaClf.estimators_[:20]:

    y_pred = estimator.predict(X_train)

    correct = y_train==y_pred

    print(correct[:10]) # print only the first 10 training samples to save space
X = data[data.columns[0:11]].values

y = data['quality'].values.astype(np.int)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)



print('X train size: ', X_train.shape)

print('X test size: ', X_test.shape)

print('y train size: ', y_train.shape)

print('y test size: ', y_test.shape)
adaReg = AdaBoostRegressor(random_state=3)

adaReg.fit(X_train, y_train)
y_pred = adaReg.predict(X_test)

plt.plot(y_test, y_pred, linestyle='', marker='o')

plt.xlabel('true values', fontsize = 16)

plt.ylabel('predicted values', fontsize = 16)

plt.show()

print('The r2_score on the test set is: ',r2_score(y_test, y_pred))
data = pd.read_csv('../input/winequality-red.csv')

data['category'] = data['quality'] >= 7 # again, binarize for classification



X = data[data.columns[0:11]].values

y = data['category'].values.astype(np.int)



# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42) 

# random_state is fixed to guarantee repeatable results for this tutorial - remove it in actual practice



print('X train size: ', X_train.shape)

print('X test size: ', X_test.shape)

print('y train size: ', y_train.shape)

print('y test size: ', y_test.shape)
clf = GradientBoostingClassifier(random_state = 27)

clf.fit(X_train, y_train)
p_pred = clf.predict_proba(X_test)[:,1]

fpr,tpr,thresholds = roc_curve(y_test, p_pred)



plt.subplots(figsize=(8,6))

plt.plot(fpr, tpr)

plt.xlim(0,1)

plt.ylim(0,1)

plt.xlabel('False positive rate', fontsize = 16)

plt.ylabel('True positive rate', fontsize = 16)

plt.show()
print("AUC is: ", auc(fpr, tpr))
tuned_parameters = {'learning_rate':[0.05,0.1,0.5,1.0], 'subsample':[0.4,0.6,0.8,1.0]}



clf = GridSearchCV(GradientBoostingClassifier(random_state = 27), tuned_parameters, cv=5, scoring = 'roc_auc')

clf.fit(X_train, y_train)
print('The best model is: ', clf.best_params_)
tuned_parameters = {'learning_rate':[0.09,0.1,0.11], 'subsample':[0.7,0.75,0.8,0.85,0.9]}



clf = GridSearchCV(GradientBoostingClassifier(random_state = 27), tuned_parameters, cv=5, scoring = 'roc_auc')

clf.fit(X_train, y_train)
print('The best model is: ', clf.best_params_)
p_pred = clf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, p_pred)



plt.subplots(figsize=(8,6))

plt.plot(fpr, tpr)

plt.xlim(0,1)

plt.ylim(0,1)

plt.xlabel('False positive rate', fontsize = 16)

plt.ylabel('True positive rate', fontsize = 16)

plt.show()
print("AUC is: ", auc(fpr, tpr))
X = data[data.columns[0:11]].values

y = data['quality'].values.astype(np.int)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)



print('X train size: ', X_train.shape)

print('X test size: ', X_test.shape)

print('y train size: ', y_train.shape)

print('y test size: ', y_test.shape)
reg = GradientBoostingRegressor(random_state = 5)

reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

plt.plot(y_test, y_pred, linestyle='', marker='o')

plt.xlabel('true values', fontsize = 16)

plt.ylabel('predicted values', fontsize = 16)

plt.show()

print('The r2_score on the test set is: ',r2_score(y_test, y_pred))
from xgboost import XGBClassifier
data = pd.read_csv('../input/winequality-red.csv')

data['category'] = data['quality'] >= 7 # again, binarize for classification



X = data[data.columns[0:11]].values

y = data['category'].values.astype(np.int)



# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42) 

# random_state is fixed to guarantee repeatable results for this tutorial - remove it in actual practice



print('X train size: ', X_train.shape)

print('X test size: ', X_test.shape)

print('y train size: ', y_train.shape)

print('y test size: ', y_test.shape)
clf = XGBClassifier(random_state = 2)

clf.fit(X_train, y_train)
p_pred = clf.predict_proba(X_test)[:,1]

fpr,tpr,thresholds = roc_curve(y_test, p_pred)



plt.subplots(figsize=(8,6))

plt.plot(fpr, tpr)

plt.xlim(0,1)

plt.ylim(0,1)

plt.xlabel('False positive rate', fontsize = 16)

plt.ylabel('True positive rate', fontsize = 16)

plt.show()
print("AUC is: ", auc(fpr, tpr))
tuned_parameters = {'gamma':[0,1,5],'reg_alpha':[0,1,5], 'reg_lambda':[0,1,5]}



clf = GridSearchCV(XGBClassifier(random_state = 2), tuned_parameters, cv=5, scoring='roc_auc')

clf.fit(X_train, y_train)
print('The best model is: ', clf.best_params_)
p_pred = clf.predict_proba(X_test)[:,1]

fpr,tpr,thresholds = roc_curve(y_test, p_pred)



plt.subplots(figsize=(8,6))

plt.plot(fpr, tpr)

plt.xlim(0,1)

plt.ylim(0,1)

plt.xlabel('False positive rate', fontsize = 16)

plt.ylabel('True positive rate', fontsize = 16)

plt.show()
print("AUC is: ", auc(fpr, tpr))
from xgboost import XGBRegressor
X = data[data.columns[0:11]].values

y = data['quality'].values.astype(np.int)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)



print('X train size: ', X_train.shape)

print('X test size: ', X_test.shape)

print('y train size: ', y_train.shape)

print('y test size: ', y_test.shape)
reg = XGBRegressor(random_state = 21)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

plt.plot(y_test, y_pred, linestyle='', marker='o')

plt.xlabel('true values', fontsize = 16)

plt.ylabel('predicted values', fontsize = 16)

plt.show()

print('The r2_score on the test set is: ',r2_score(y_test, y_pred))
tuned_parameters = {'gamma':[0,1,5], 'reg_lambda': [1,5,10], 'reg_alpha':[0,1,5], 'subsample': [0.6,0.8,1.0]}



reg = GridSearchCV(XGBRegressor(random_state = 21, n_estimators=500), tuned_parameters, cv=5, scoring='r2')

reg.fit(X_train, y_train)
print('The best model is: ', reg.best_params_)
y_pred = reg.predict(X_test)

plt.plot(y_test, y_pred, linestyle='', marker='o')

plt.xlabel('true values', fontsize = 16)

plt.ylabel('predicted values', fontsize = 16)

plt.show()

print('The r2_score on the test set is: ',r2_score(y_test, y_pred))