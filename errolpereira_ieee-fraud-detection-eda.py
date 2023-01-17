# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

import xgboost as xgb

import lightgbm as lgb

from sklearn.metrics import accuracy_score, classification_report

from scikitplot.metrics import plot_confusion_matrix

from scikitplot.classifiers import plot_feature_importances

import gc
PATH = '/kaggle/input/ieee-fraud-detection/'
#Reading the datasets

train_identity = pd.read_csv(PATH + 'train_identity.csv', index_col='TransactionID')

train_transaction = pd.read_csv(PATH + 'train_transaction.csv', index_col='TransactionID')



test_identity = pd.read_csv(PATH + 'test_identity.csv', index_col='TransactionID')

test_transaction = pd.read_csv(PATH + 'test_transaction.csv', index_col='TransactionID')



sample_submission = pd.read_csv(PATH + 'sample_submission.csv', index_col='TransactionID')
#creating train and test dataframe.

train = pd.merge(train_transaction, train_identity, how='left', left_index=True, right_index=True)

test = pd.merge(test_transaction, test_identity, how='left', left_index=True, right_index=True)
#printing the shape

print(train.shape)

print(test.shape)
#creating X and y variable

X = train.drop('isFraud', axis=1)

y = train.isFraud.copy()
#deleting unwanted dataframes

del train_identity, train_transaction, test_identity, test_transaction, train

gc.collect()
#filling null values

X = X.fillna(-999)

test = test.fillna(-999)
# Label Encoding

for f in X.columns:

    if X[f].dtype=='object' or test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X[f].values) + list(test[f].values))

        X[f] = lbl.transform(list(X[f].values))

        test[f] = lbl.transform(list(test[f].values))   
#splitting training dataset into train and test

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
gc.collect()
# #XGBoost Model.

# #initializing the model

# clf = xgb.XGBClassifier(

#     n_estimators=500,

#     max_depth=9,

#     learning_rate=0.05,

#     subsample=0.9,

#     colsample_bytree=0.9,

#     missing=-999,

#     random_state=2019,

#     tree_method='gpu_hist'  # THE MAGICAL PARAMETER

# )



# #training on training data.

# % time clf.fit(X_train, y_train)



# #prdicting on testing data.

# y_pred = clf.predict(X_test)



# #evaluation.

# print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))

# print(classification_report(y_test, y_pred))

# plot_confusion_matrix(y_test, y_pred)
#Light Gradient Model.

#initializing the model

clf = lgb.LGBMClassifier(

    max_depth=9,

    num_leaves=96,

    n_estimators=350,

    subsample_for_bin=1500,

    subsample=0.9,

    colsample_bytree=0.9,

)



#training on training data.

%time clf.fit(X_train, y_train)



#prdicting on testing data.

y_pred = clf.predict(X_test)



#evaluation.

print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))

print(classification_report(y_test, y_pred))

plot_confusion_matrix(y_test, y_pred)
gc.collect()
clf.fit(X, y)
sample_submission['isFraud'] = clf.predict_proba(test)[:,1]

sample_submission.to_csv('simple_xgboost.csv')
del X_train, y_train, X_test, y_test

gc.collect()
gc.collect()