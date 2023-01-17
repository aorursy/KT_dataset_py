# Load packages

import numpy as np   # linear algebra

import pandas as pd  # data processing



# Input data files are available in the "../input/" directory.

# List all files under the input directory

#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



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



# Get target, remove it from training data

y = trn['isFraud']

trn = trn.drop('isFraud', axis=1)
# If pct of missing data in a col > pct_thresh then remove col 

pct_thresh = 0.45

# 47% = 228 dropped cols

# 46% = 228 dropped cols

# 45% = 231 dropped cols

# 40% = 232 dropped cols



numRows = trn.shape[0]

col_list = trn.columns.values.tolist()



# Determine which cols to keep

new_cols = []

for col in col_list:

    missing_ratio = trn[trn[col].isnull()].shape[0] / numRows

    if missing_ratio < pct_thresh:

        new_cols.append(col)

        

print('Num cols dropped:', len(col_list) - len(new_cols))

print('Remaining cols:', len(new_cols))
# Update train and test dataframes.

trn = trn[new_cols]

tst = tst[new_cols]
# Get list of categorical variables

cat_cols = (trn.dtypes == 'object')

cat_cols = cat_cols[cat_cols].index.tolist()

print(cat_cols)
# Imputation - Replace missing values with 'missingP_email'

trn.loc[ trn['P_emaildomain'].isnull(), 'P_emaildomain'] = 'missingP_email'

tst.loc[ tst['P_emaildomain'].isnull(), 'P_emaildomain'] = 'missingP_email'
# Change 'gmail' to 'gmail.com'

trn.loc[trn.P_emaildomain == 'gmail', 'P_emaildomain'] = 'gmail.com'

tst.loc[tst.P_emaildomain == 'gmail', 'P_emaildomain'] = 'gmail.com'
# Find all domains that have no fraud

fraud_count = trn.P_emaildomain.value_counts().to_frame()

fraud_count['frauds'] = 0

fraud_count['pct'] = 0



for domain in fraud_count.index:

    num_frauds = trn['P_emaildomain'].loc[(trn.P_emaildomain == domain) & (y == True)].count()

    pct = num_frauds / fraud_count.loc[domain, 'P_emaildomain']

    fraud_count.loc[domain, 'frauds'] = num_frauds

    fraud_count.loc[domain, 'pct'] = pct

    

no_fraud = fraud_count.loc[fraud_count['frauds'] == 0].index.to_list() 



# Rename all no-fraud domains to 'zero_fraud'

for domain in no_fraud:

    trn.loc[trn.P_emaildomain == domain, 'P_emaildomain'] = 'zero_fraud'

    tst.loc[tst.P_emaildomain == domain, 'P_emaildomain'] = 'zero_fraud'    
#print(trn.columns)

print(len(trn.columns))
#############

# Count occurances of each email domain. Store results in dataframe.

#fraud_count = trn.P_emaildomain.value_counts().to_frame()



# Determine num frauds and pct frauds for each domain. 

# Replace each email domain with pct fraud

#fraud_count['frauds'] = 0

#fraud_count['pct'] = 0



# Imputation: add new column for pct fraud for each email domain

#trn.loc['P_email_pct_fraud'] = 0

#tst.loc['P_email_pct_fraud'] = 0



#for domain in fraud_count.index:

#    trn.loc[trn.P_emaildomain == domain, 'P_email_pct_fraud'] = fraud_count.loc[domain, 'pct']

#    tst.loc[tst.P_emaildomain == domain, 'P_email_pct_fraud'] = fraud_count.loc[domain, 'pct']



# Note that tst has an email domain 'scranton.edu' that does not appear in 

# the trn data. Have to deal with this case seperately

#tst.loc[tst.P_emaildomain == 'scranton.edu', 'P_email_pct_fraud'] = fraud_count.pct.mean()



# Drop P_emaildomain 

#trn = trn.drop('P_emaildomain', axis=1)

#tst = tst.drop('P_emaildomain', axis=1)



#no_fraud = fraud_count.loc[fraud_count['frauds'] == 0].index.to_list() 



# Rename all no-fraud domains to 'zero_fraud'

#for domain in no_fraud:

#    trn.loc[trn.P_emaildomain == domain, 'P_emaildomain'] = 'zero_fraud'

#    tst.loc[tst.P_emaildomain == domain, 'P_emaildomain'] = 'zero_fraud'
# 'ProductCD' will take a little more massaging to process. 

# For now let's drop it. Can come back to later.

trn = trn.drop(['ProductCD'], axis=1)

tst = tst.drop(['ProductCD'], axis=1)



# Replace missing entries (NaN) with marker string

trn.loc[trn['card4'].isnull(), ['card4']] = 'missing4'

trn.loc[trn['card6'].isnull(), ['card6']] = 'missing6'

#trn.loc[trn['M1'].isnull(), ['M1']] = 'missingM1'

#trn.loc[trn['M2'].isnull(), ['M2']] = 'missingM2'

#trn.loc[trn['M3'].isnull(), ['M3']] = 'missingM3'

trn.loc[trn['M6'].isnull(), ['M6']] = 'missingM6'

tst.loc[tst['card4'].isnull(), ['card4']] = 'missing4'

tst.loc[tst['card6'].isnull(), ['card6']] = 'missing6'

#tst.loc[tst['M1'].isnull(), ['M1']] = 'missingM1'

#tst.loc[tst['M2'].isnull(), ['M2']] = 'missingM2'

#tst.loc[tst['M3'].isnull(), ['M3']] = 'missingM3'

tst.loc[tst['M6'].isnull(), ['M6']] = 'missingM6'



# trn['card6'] contains the category 'debit or credit'. 

# As these are the only two possiblities, this info is not helpful. 

# Change 'debit or credit' to 'missing'

# tst['card6'] does not have this category

trn.loc[trn['card6'] == 'debit or credit', ['card6']] = 'missing6'
# Update list of categorical variables

cat_cols = (trn.dtypes == 'object')

cat_cols = cat_cols[cat_cols].index.tolist()



dummy_cols = []

for col in cat_cols:

    print(col)

    # Process training data

    # Get dummies, add to trn

    dummy = pd.get_dummies(trn[col])

    dummy_cols.extend(dummy.columns.tolist())

    trn = pd.concat([trn, dummy], axis = 1)

    # Now process test data. As mentioned above, tst has an extra category in 

    # P_emaildomain: 'scranton.edu'. But there are only 2 instances. This will

    # will result in an extra dummy col in the tst dataframe.

    dummy = pd.get_dummies(tst[col])

    tst = pd.concat([tst, dummy], axis = 1)



# Now perform an inner join on the two dataframes. This (hopefully) removes

# cols that do not appear in both dataframes. In particular, this should remove

# the dummy col for 'scranton.edu' from the tst dataframe.

trn, tst = trn.align(tst, join='inner', axis=1)



# Drop categorical columns

trn = trn.drop(cat_cols, axis=1)

tst = tst.drop(cat_cols, axis=1)
print(len(trn.columns))
# Get all cols in a list

cols = trn.columns.values.tolist()



# Remove dummy cols from list, since they will not have missing values.

for col in dummy_cols:

    print(col)

    cols.remove(col)

    

# Perform imputation. This will run through the one-hot cols as well. It's probably best to skip those,

# but shouldn't cause a problem if you don't, as they should not be missing data.



for col in cols:

    #print(col)

    mean = trn[col].mean()

    trn[col] = trn[col].fillna(mean)

    tst[col] = tst[col].fillna(mean)
## Create Model
# Make sure columns are in the same order for train and test

cols = trn.columns.values.tolist()

tst = tst[cols]
# Let's try XGBoost

import xgboost as xgb



# Specify Model: Random Forest

#from sklearn.ensemble import RandomForestRegressor

#seed = 1

#clf = RandomForestRegressor(n_estimators=250, max_depth=30, random_state = seed)



clf = xgb.XGBClassifier(

    bagging_fraction = 0.9,

    objective = 'binary:logistic', # Did not change results

    n_estimators = 300,

    max_depth=16,

    learning_rate=0.014,

    subsample=0.5,

    colsample_bytree = 0.75,

    num_leaves = 220,    # Did not change results

    #missing=-999,

    random_state=1,

    tree_method='exact'  # Did not change results # THE MAGICAL PARAMETER

)



clf.fit(trn, y)



# Make predictions

predictions = clf.predict(tst)
submission = pd.DataFrame(index = tst.index)

submission['isFraud'] = predictions

submission.to_csv('nov14.csv')
#trn['P_emaildomain'].isnull().sum()

#myList = list(set(tst.P_emaildomain.value_counts().index.to_list()) - 

#              set(trn.P_emaildomain.value_counts().index.to_list()))

#print(myList)

#email.sort_values(by=['pct','frauds'])



email_counts = trn.P_emaildomain.value_counts()

email = email_counts.to_frame()

#print(email_counts)

email['frauds'] = 0

email['pct'] = 0

email

for domain in email.index:

    numFraud = y.loc[trn['P_emaildomain'] == domain].sum() 

    email.loc[email.index == domain, 'frauds'] = numFraud

#totFraud = email.frauds.sum()

email['pct'] = 100 * email['frauds'] / email['P_emaildomain']

email
#tst['P_emaildomain'].loc[tst.P_emaildomain == 'gmail.com']
#cols = trn.columns.values.tolist()

#list = list(set(dummy_cols) - set(cols))

#print(list)

#trn.P_emaildomain



cat_cols = (trn.dtypes == 'object')

cat_cols = cat_cols[cat_cols].index.tolist()

cat_cols

#trn.dtypes

#trn.astype({'P_emaildomain': 'float64'}).dtypes