# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import sklearn

import pandas

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



training_data = pd.read_csv('../input/train.csv') #training data frame

train_y = training_data['Survived'] #get y before removing this column

training_data = training_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

#training_data = training_data.drop([61,829]) #get rid of training rows with no embarked

mean_age_training = training_data['Age'].mean()

training_data['Age']=training_data['Age'].fillna(mean_age_training)

training_data['Embarked']=training_data['Embarked'].fillna('S')

#print(training_data.isnull().any()) check for NaNs



test_data = pd.read_csv('../input/test.csv')

test_data = test_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

mean_age_test = test_data['Age'].mean()

mean_fare_test = test_data['Fare'].mean()

test_data['Age']=test_data['Age'].fillna(mean_age_test)

test_data['Fare']=test_data['Fare'].fillna(mean_fare_test)

#print(test_data.isnull().any())



#print (training_data.loc[829,:])



#One-hot encoding for sex and embark

le_Sex = LabelEncoder()

le_Embarked = LabelEncoder()

Sex_ohe = OneHotEncoder()

Embarked_ohe = OneHotEncoder()

Class_ohe = OneHotEncoder()



training_data['Sex_encoded'] = le_Sex.fit_transform(training_data.Sex)

training_data['Embarked_encoded'] = le_Embarked.fit_transform(training_data.Embarked)

#print (training_data) ordering of one-hot columns comes from default set here



X = Sex_ohe.fit_transform(training_data.Sex_encoded.values.reshape(-1,1)).toarray()

X2 = Embarked_ohe.fit_transform(training_data.Embarked_encoded.values.reshape(-1,1)).toarray()

X3 = Class_ohe.fit_transform(training_data.Pclass.values.reshape(-1,1)).toarray()

dfOneHot = pd.DataFrame(X, columns = ['Female','Male'])

training_data = pd.concat([training_data, dfOneHot], axis=1)

dfOneHot = pd.DataFrame(X2, columns = ['C','Q','S'])

training_data = pd.concat([training_data, dfOneHot], axis=1)

dfOneHot = pd.DataFrame(X3, columns = ['Class1','Class2','Class3'])

training_data = pd.concat([training_data, dfOneHot], axis=1)

training_data = training_data.drop(columns = ['Sex','Embarked','Pclass'])

#training_data = training_data.drop([61,829]) still confused - does dropping affect row ordering??

#print (training_data)





test_data['Sex_encoded'] = le_Sex.fit_transform(test_data.Sex)

test_data['Embarked_encoded'] = le_Embarked.fit_transform(test_data.Embarked)

X = Sex_ohe.fit_transform(test_data.Sex_encoded.values.reshape(-1,1)).toarray()

X2 = Embarked_ohe.fit_transform(test_data.Embarked_encoded.values.reshape(-1,1)).toarray()

X3 = Class_ohe.fit_transform(test_data.Pclass.values.reshape(-1,1)).toarray()

dfOneHot = pd.DataFrame(X, columns = ['Female','Male'])

test_data = pd.concat([test_data, dfOneHot], axis=1)

dfOneHot = pd.DataFrame(X2, columns = ['C','Q','S'])

test_data = pd.concat([test_data, dfOneHot], axis=1)

dfOneHot = pd.DataFrame(X3, columns = ['Class1','Class2','Class3'])

test_data = pd.concat([test_data, dfOneHot], axis=1)

test_data = test_data.drop(columns = ['Sex','Embarked','Pclass'])



from sklearn.preprocessing import StandardScaler

StandardScaler(copy=True, with_mean=True, with_std=True)

scaler1 = StandardScaler()

scaler2 = StandardScaler()

scaler1.fit(training_data)

scaler2.fit(test_data)



train_X = scaler1.transform(training_data.values)

test_X = scaler2.transform(test_data.values) #convert to a numpy array



print (train_X.shape)

print (test_X.shape)
from sklearn.model_selection import train_test_split, StratifiedKFold 

from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Lasso

from sklearn import svm

import random

import catboost

random_seed = 213

import xgboost as xgb

np.random.seed(random_seed)

"""

#log reg tuning

clf = LogisticRegression(class_weight='balanced', solver='liblinear',max_iter=10000)

penalty = ['l1', 'l2']

# Create regularization hyperparameter space

C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

print (C)

# Create hyperparameter options

hyperparameters = dict(C=C, penalty=penalty)



#SVM tuning

parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}

svc = svm.SVC(gamma="scale")



gs = GridSearchCV(clf, hyperparameters, cv=5, verbose = 1)

gs.fit(train_X, train_y)

best_score = gs.best_score_

best_estimator = gs.best_estimator_

print('Best Penalty:', gs.best_estimator_.get_params()['penalty'])

print('Best C:', gs.best_estimator_.get_params()['C'])

"""



#best_estimator = svm.SVC(C = 1, kernel = 'rbf', gamma="scale")

#best_estimator = catboost.CatBoostClassifier(iterations=100, random_seed=0, verbose=False)



best_estimator = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.65, gamma=2, learning_rate=0.3, max_delta_step=1,

       max_depth=4, min_child_weight=2, missing=None, n_estimators=280,

       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,

       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

       silent=True, subsample=1)



N_SPLITS = 10

splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True).split(train_X, train_y))



y_preds = []

for idx, (train_idx, val_idx) in enumerate(splits):

    print("Beginning fold {}".format(idx+1))

    X_train, y_train, X_val, y_val = train_X[train_idx], train_y[train_idx], train_X[val_idx], train_y[val_idx]

    #XX = np.vstack([np.ones((X_train.shape[0],)), X_train.T]).T

    best_estimator.fit(X_train,y_train)

    score = roc_auc_score(y_val, best_estimator.predict(X_val))

    print(score)     

    #y_preds.append(best_estimator.predict_proba(test_X)[:,1].reshape((19750,1)))

    y_preds.append(best_estimator.predict(test_X).reshape((418,1)))

y_preds = np.concatenate(y_preds, axis=1)

y_preds.shape
subs = pd.read_csv('../input/gender_submission.csv')

mean_preds = y_preds.mean(axis=1)

mean_preds[mean_preds < 0.5] = 0

mean_preds[mean_preds >= 0.5] = 1

subs['Survived'] = mean_preds.astype(int)

subs.to_csv('titanic_xgboost.csv', index=False)