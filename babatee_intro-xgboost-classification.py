# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from sklearn.cross_validation import train_test_split



# Input data files are available in the "../input/" directory.b

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#load Dataset

data = pd.read_csv('../input/diabetes.csv')

data.head()
data_full = data.copy()

X_data = data_full.drop('Outcome', axis=1)

y = data_full.Outcome
X_data.head()
#Split the dataset into train and Test

seed = 7

test_size = 0.3

X_trian, X_test, y_train, y_test = train_test_split(X_data, y, test_size=test_size, random_state=seed)
#Train the XGboost Model for Classification

model1 = xgb.XGBClassifier()

model2 = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)



train_model1 = model1.fit(X_trian, y_train)

train_model2 = model2.fit(X_trian, y_train)
#prediction and Classification Report

from sklearn.metrics import classification_report



pred1 = train_model1.predict(X_test)

pred2 = train_model2.predict(X_test)



print('Model 1 XGboost Report %r' % (classification_report(y_test, pred1)))

print('Model 2 XGboost Report %r' % (classification_report(y_test, pred2)))
#Let's use accuracy score

from sklearn.metrics import accuracy_score



print("Accuracy for model 1: %.2f" % (accuracy_score(y_test, pred1) * 100))

print("Accuracy for model 2: %.2f" % (accuracy_score(y_test, pred2) * 100))
#Let's do a little Gridsearch, Hyperparameter Tunning

model3 = xgb.XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)
train_model3 = model3.fit(X_trian, y_train)

pred3 = train_model3.predict(X_test)

print("Accuracy for model 3: %.2f" % (accuracy_score(y_test, pred3) * 100))
from sklearn.model_selection import GridSearchCV



param_test = {

 'max_depth':[4,5,6],

 'min_child_weight':[4,5,6]

}

gsearch = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,

 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)



train_model4 = gsearch.fit(X_trian, y_train)

pred4 = train_model4.predict(X_test)

print("Accuracy for model 4: %.2f" % (accuracy_score(y_test, pred4) * 100))
param_test2b = {

 'min_child_weight':[6,8,10,12]

}

gsearch2b = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=4,

 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)



train_model5 = gsearch2b.fit(X_trian, y_train)

pred5 = train_model5.predict(X_test)

print("Accuracy for model 5: %.2f" % (accuracy_score(y_test, pred5) * 100))
#Tune Gamma

param_test3 = {

 'gamma':[i/10.0 for i in range(0,5)]

}

gsearch3 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,

 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)



train_model6 = gsearch3.fit(X_trian, y_train)

pred6 = train_model6.predict(X_test)

print("Accuracy for model 6: %.2f" % (accuracy_score(y_test, pred6) * 100))
xgb2 = xgb.XGBClassifier(

 learning_rate =0.7,

 n_estimators=1000,

 max_depth=4,

 min_child_weight=6,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)



train_model7 = xgb2.fit(X_trian, y_train)

pred7 = train_model7.predict(X_test)

print("Accuracy for model 7: %.2f" % (accuracy_score(y_test, pred7) * 100))
#Let's train a fast RandomForest on the dataset

from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier()

rfc_model = rfc.fit(X_trian, y_train)

pred8 = rfc_model.predict(X_test)

print("Accuracy for Random Forest Model: %.2f" % (accuracy_score(y_test, pred8) * 100))