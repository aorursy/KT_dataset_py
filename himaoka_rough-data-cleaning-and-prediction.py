import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import gc

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.feature_selection import SelectKBest, f_regression, f_classif

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Set pandas data display option

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 500)
train_identity = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")

train_transaction = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")



test_identity = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")

test_transaction = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")



sample_submission = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")



# All data

data_identity = train_identity.append(test_identity, sort=False)

data_transaction = train_transaction.append(test_transaction, sort=False)
train_transaction.info()
test_transaction.info()
data_identity.info()
data_identity.head(5)
data_transaction.info()
data_transaction.head(5)
sns.countplot('isFraud', data=data_transaction)
data_transaction[data_transaction['TransactionID'].isin(data_identity['TransactionID'])]
# Just merge Identity and Transaction data on TransactionID

data = pd.merge(data_transaction, data_identity, on='TransactionID', how='left')



del data_identity

del data_transaction

del train_identity

del train_transaction

del test_identity

del test_transaction



gc.collect()
data
types = pd.DataFrame(data.dtypes).rename(columns={0: 'type'}).sort_values(by=['type'], ascending=False)



types
# Check missing values

def check_missing(df):

    null_val = df.isnull().sum()

    percent = 100 * df.isnull().sum()/len(df)

    missing_table = pd.concat([null_val, percent], axis=1)

    col = missing_table.rename(columns = {0 : 'Num', 1 : 'Rate'})

    return col



# Display columns missing values are under 1%.

print("Data #"+str(len(data)))

cols = check_missing(data)

types = types.join(cols).sort_values(by="Rate", ascending=False)



types
cols_missing = types[types['Rate'] > 80].index.values.tolist()



# Drop more than 50% missing variables

types.drop(cols_missing, axis=0, inplace = True)

data.drop(cols_missing, axis=1, inplace = True)
objects = types[types['type'] == 'object'].index.values.tolist()

float64s = types[types['type'] == 'float64'].index.values.tolist()

int64s = types[types['type'] == 'int64'].index.values.tolist()



int64s.remove('TransactionID')

float64s.remove('isFraud')
for v in objects:

    # Fill NaN with mode

    data[v] = data[v].fillna(data[v].mode()[0])

    # One-Hot Encoding

    # data = pd.get_dummies(data, columns=[v], drop_first=True)

    # Categorize

    data[v] = pd.factorize(data[v])[0]



ss = StandardScaler()



for v in int64s:

    # Fill NaN with mean

    data[v] = data[v].fillna(data[v].mean())

    # Standardize values

    data[v] = ss.fit_transform(data[[v]])



for v in float64s:

    # Fill NaN with mean

    data[v] = data[v].fillna(data[v].mean())

    # Standardize values

    data[v] = ss.fit_transform(data[[v]])
# Data example

data.sample(n=10)
# Set data

train = data[:590540]

test  = data[590540:]

train_target = train[:590540]["isFraud"].values
possible_features = train.columns.copy().drop('TransactionID').drop('isFraud')



train_sample = train.sample(100000)



# Check feature importances

selector = SelectKBest(f_classif, len(possible_features))

selector.fit(train_sample[possible_features], train_sample['isFraud'])

scores = -np.log10(selector.pvalues_)

indices = np.argsort(scores)[::-1]

print('Feature importances:')

importances = pd.DataFrame([], columns=['col','importance'])

for i in range(len(scores)):

    #print('%.2f %s' % (scores[indices[i]], possible_features[indices[i]]))

    addRow = pd.DataFrame([possible_features[indices[i]],scores[indices[i]]], index=importances.columns).T

    importances = importances.append(addRow)



importances.sort_values(by='importance', ascending=False)
fparams_df = importances[importances['importance'] > 10]

fparams = fparams_df['col'].values

fparams
# Get params

train_features = train[fparams].values

test_features  = test[fparams].values



# Number of Cross Validation Split

CV_SPLIT_NUM = 4



# Params for RandomForestClassifier

rfgs_parameters = {

    'n_estimators': [300],

    'max_depth'   : [2,3,4],

    'max_features': [2,3,4],

    "min_samples_split": [2,3,4],

    "min_samples_leaf":  [2,3,4]

}



# rfc_cv = GridSearchCV(RandomForestClassifier(), rfgs_parameters, cv=CV_SPLIT_NUM)

# rfc_cv.fit(train_features, train_target)

# model = rfc_cv.best_estimator_

# print("RFC GridSearch score: "+str(rfc_cv.best_score_))

# print("RFC GridSearch params: ")

# print(rfc_cv.best_params_)



rfc = RandomForestClassifier(n_estimators=1000, max_depth=12, max_features='sqrt')

rfc.fit(train_features, train_target)

model = rfc
# Predict and output to csv

isFraud = model.predict(test_features)

pred = pd.DataFrame(test['TransactionID'].copy())

pred['isFraud'] = isFraud.astype(int)

pred.to_csv("../working/submission.csv", index = False)