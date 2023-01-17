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
# Path to data files

path = '../input/ieee-fraud-detection/'



# Load transaction data

train_transaction = pd.read_csv(path + 'train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv(path + 'test_transaction.csv', index_col='TransactionID')



# Load identity data

train_identity = pd.read_csv(path + 'train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv(path + 'test_identity.csv', index_col='TransactionID')



# These merges will keep the data in the left dataframe. Data from the right 

# dataframe will only be kept if the index appears in the left dataframe.

trn = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

tst = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



# Delete dataframes no longer needed to free up RAM

del train_transaction, train_identity, test_transaction, test_identity
# Parameters

pct_thresh = 0.2



numRows = trn.shape[0]

col_list = trn.columns.values.tolist()



# Initialize list to hold cols to drop

drop_cols = []

for col in col_list:

    missing_ratio = trn[trn[col].isnull()].shape[0] / numRows

    if missing_ratio > pct_thresh:

        drop_cols.append(col)

        

# Drop columns

for col in drop_cols:

    col_list.remove(col)

    

print(len(drop_cols))

#print(drop_cols)



# Update train and test dataframes.

trn = trn[col_list]

tst = tst[col_list[1:]]
# 'ProductCD' and 'P_emaildomain' will take a little more massaging to process. 

# For now let's drop them. We can come back to them later.

trn = trn.drop(['ProductCD', 'P_emaildomain'], axis=1)

tst = tst.drop(['ProductCD', 'P_emaildomain'], axis=1)



# Replace missing entries (NaN) with string 'missing'

trn.loc[trn['card4'].isnull(), ['card4']] = 'missing4'

trn.loc[trn['card6'].isnull(), ['card6']] = 'missing6'

tst.loc[tst['card4'].isnull(), ['card4']] = 'missing4'

tst.loc[tst['card6'].isnull(), ['card6']] = 'missing6'
# trn['card6'] contains the category 'debit or credit'. 

# As these are the only two possiblities, this info is not helpful. 

# Change 'debit or credit' to 'missing'

# tst['card6'] does not have this category

trn.loc[trn['card6'] == 'debit or credit', ['card6']] = 'missing6'
# Let's change 'charge card' entry in the 'card6' field to 'credit'

trn.loc[trn['card6'] == 'charge card', ['card6']] = 'credit'

tst.loc[tst['card6'] == 'charge card', ['card6']] = 'credit'
#trn['card6'].value_counts()
#trn.head()

#trn.loc[trn['debit or credit'] == 1]

#ts
from sklearn.preprocessing import OneHotEncoder



# Get list of categorical variables

cat_cols = (trn.dtypes == 'object')

cat_cols = list(cat_cols[cat_cols].index)

print(cat_cols)



# Apply one-hot encoder to each column with categorical data

# handle_unknows = 'ignore' will result in a row of 0s when fitting and a 

# previously unseen category is encountered.

# sparse = 'False' places output in a matrix (I think)

hot1_encoder = OneHotEncoder(handle_unknown='ignore', sparse='False')

hot1_cols_trn = pd.DataFrame(hot1_encoder.fit_transform(trn[cat_cols]).toarray())

hot1_cols_tst = pd.DataFrame(hot1_encoder.transform(tst[cat_cols]).toarray())



# Add index

hot1_cols_trn.index = trn.index

hot1_cols_tst.index = tst.index



# Drop categorical cols, will replace w/ one-hot cols

trn = trn.drop(cat_cols, axis=1)

tst = tst.drop(cat_cols, axis=1)



# Add one-hot cols

trn = pd.concat([trn, hot1_cols_trn], axis=1)

tst = pd.concat([tst, hot1_cols_tst], axis=1)
#trn[cat_cols].head()

#hot1_cols_trn.head()
# Pull out the target

y = trn['isFraud']

trn = trn.drop(columns = ['isFraud'])



# Get all cols in a list

cols = trn.columns.values.tolist()



# Perform imputation. This will run through the one-hot cols as well. It's probably best to skip those,

# but shouldn't cause a problem if you don't, as they should not be missing data.



for col in cols:

    #print(col)

    mean = trn[col].mean()

    trn[col] = trn[col].fillna(mean)

    tst[col] = tst[col].fillna(mean)
# Make sure columns are in the same order for train and test

cols = trn.columns.values.tolist()

tst = tst[cols]
# Let's try XGBoost

import xgboost as xgb



clf = xgb.XGBClassifier(

    n_estimators=300,

    max_depth=15,

    learning_rate=0.02,

    subsample=0.5,

    colsample_bytree=0.9,

    missing=-999,

    random_state=1,

    tree_method='exact'  # THE MAGICAL PARAMETER

)



clf.fit(trn, y)



# Make predictions

predictions = clf.predict(tst)
#print(trn.columns.to_list())
# Specify Model

#from sklearn.ensemble import RandomForestRegressor

#seed = 1

#rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state = seed)



# Fit Model (Can take hours to run)

#rf_model.fit(trn, y)



# Make predictions

#predictions = rf_model.predict(tst)
submission = pd.DataFrame(index = tst.index)

submission['isFraud'] = predictions

submission.to_csv('nov10.csv')
#trn['card2'].loc[trn['card2'].isnull()]#.sum()