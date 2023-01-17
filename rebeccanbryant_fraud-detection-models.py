# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import all of these programs



import os



import numpy as np

import pandas as pd

from sklearn import preprocessing

import xgboost as xgb



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



# Path to data files

path = '../input/ieee-fraud-detection/'



# Transaction data

train_transaction = pd.read_csv(path + 'train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv(path + 'test_transaction.csv', index_col='TransactionID')



#Identity data

train_identity = pd.read_csv(path + 'train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv(path + 'test_identity.csv', index_col='TransactionID')
# What does our data look like?



#print(train_transaction.shape)

#print(train_identity.shape)

#print(train_transaction[train_transaction['isFraud'] == 1].shape)

##train_transaction.columns #<- this doesn't give us all of our columns

#print(train_transaction.columns.values)
# Merge our training and test dataframes to create two new dataframes of training and test data

trn = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

tst = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



# Now that we merged our dataframes, delete the originals so that we aren't eating up RAM unnecessarily.

del(train_identity, train_transaction, test_identity, test_transaction)

#print(train_transaction[train_transaction['TransactionID']])
# What do our merged dataframes look like?

# What are our columns now?



print(trn.shape)

print(tst.shape)

#print(trn.columns.values)
#trn = trn.iloc[:100000]

#tst = tst.iloc[:1000]
#print(trn.shape)

#print(tst.shape)
#print(trn.head())

#print(tst.head())
maxPercent = .35

num_rows = trn.shape[0]

drop_threshhold = num_rows*maxPercent

trn = trn.dropna(thresh = drop_threshhold, axis = 1)

trnCols = trn.columns

#trnCols = trnCols.drop('isFraud')

trn = trn[trnCols]

#tst = tst[trnCols]

print(trn.columns.values)
features = trn.columns

print(features)

#catCols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'DeviceType', 'DeviceInfo', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']

#catCols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']

#catCols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'DeviceType', 'DeviceInfo', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']

catCols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',]

for col in catCols:

    features = features.drop(col)

print(features)
print(trn.isna().sum().sum())

print(tst.isna().sum().sum())
trnNulls = trn.isna().sum()

print(trnNulls[trnNulls > 0])



print("  ")



#tstNulls = tst.isnull().sum()

#print(tstNulls[tstNulls > 0])



trn.fillna(trn.mean(), inplace = True)

print(trn.isna().sum().sum())
trainDF = trn.fillna(-999)

print(trainDF.isna().sum().sum())

testDF = tst.fillna(-999)

print(testDF.isna().sum().sum())
y_train = trainDF['isFraud'].copy()

features = features.drop('isFraud')

X_train = trainDF[features]

#X_train.append(le_addr1)

X_test = testDF[features]
from xgboost import XGBClassifier

xgbModel = XGBClassifier( n_estimators=250,

    max_depth=12,

    subsample=0.4,

    colsample_bytree=0.5, colsample_bylevel=0.5, colsample_bynode=0.5,

    random_state=2)

xgbModel.fit(X_train, y_train)

preds = xgbModel.predict(X_test)

print(preds)
fraudModel = RandomForestRegressor(random_state = 1, n_estimators = 300, max_depth = 15)

fraudModel.fit(X_train, y_train)

predictions = fraudModel.predict(X_test)

print(predictions)
n = 4

frcn_1 = 1/n

frcn_2 = 1-frcn_1



avgPreds = frcn_1*preds + frcn_2*predictions

print(avgPreds)
path = '../input/ieee-fraud-detection/'



predictions_submission = pd.read_csv(path + 'sample_submission.csv', index_col='TransactionID')

print(predictions_submission.head())

print(predictions_submission['isFraud'].shape)

#print(predictions.shape)

#print(preds.shape)

print(avgPreds.shape)

#predictions_submission['isFraud'] = predictions

#predictions_submission['isFraud'] = preds

predictions_submission['isFraud'] = avgPreds



pd.DataFrame(predictions_submission).to_csv('predictions_submission.csv', index = True)