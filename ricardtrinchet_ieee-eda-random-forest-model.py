import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



from sklearn import preprocessing

import xgboost as xgb



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics



import matplotlib.pyplot as plt

import seaborn as sns





from sklearn.metrics import roc_auc_score

import time
# data load

train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')



sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')
## Function to reduce the DF size: 

# --- taken from https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print(train.shape)

print(test.shape)
## REducing memory

train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
del train_transaction, train_identity, test_transaction, test_identity
# subselect variables: :::::::::::::::: manually ::::::::::::::

variables  = ['TransactionDT','TransactionAmt','card1','card2', 'card3', 'card5', 'addr1', 'addr2', 'dist1',

                       'C1', 'C2','C3', 'C4','C5', 'C6','C7', 'C8','C9', 'C10', 'C11', 'C12','C13', 'C14',

              'DeviceType', 'isFraud']



train =  train[variables]

variables.remove('isFraud')

test = test[variables]



print(train.shape)



print(test.shape)
# alternative encoding



Fraud = train['isFraud'].copy()

train_df = train.drop('isFraud', axis=1).copy()

df = pd.concat([test, train_df])

traindex = train_df.index

testdex = test.index

df.head(4)
#df_dummy = pd.get_dummies(df)



# Scaling between -1 and 1. Good practice for continuous variables.

from sklearn import preprocessing

continuous_features = ['TransactionDT','TransactionAmt','card1','card2', 'card3', 'card5', 'addr1', 'addr2', 'dist1',

                       'C1', 'C2','C3', 'C4','C5', 'C6','C7', 'C8','C9', 'C10', 'C11', 'C12','C13', 'C14']

for col in continuous_features:

    transf = df[col].values.reshape(-1,1)

    scaler = preprocessing.StandardScaler().fit(transf)

    df[col] = scaler.transform(transf)
# Finish Pre-Processing

# Dummmy Variables (One Hot Encoding)

df = pd.get_dummies(df, columns=['DeviceType'])

df.columns



#df = df.drop(['Cabin', 'Name', 'Ticket'], axis = 1)

# Now that pre-processing is complete, split data into train/test again.

train_df = df.loc[traindex, :]

train_df['isFraud'] = Fraud

test_df = df.loc[testdex, :]



train_df = df.loc[traindex, :]

train_df['isFraud'] = Fraud

test_df = df.loc[testdex, :]



### 

X_train = train_df.drop(['isFraud'], axis = 1)

y_train = train_df['isFraud']

X_test = test_df





# fill NA values

X_train = X_train.fillna(-999)

X_test = X_test.fillna(-999)
threshold = 0.98

    

# Absolute value correlation matrix

corr_matrix = train[train['isFraud'].notnull()].corr().abs()



# Getting the upper triangle of correlations

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]



print('There are %d columns to remove.' % (len(to_drop)))

train = train.drop(columns = to_drop)

test = test.drop(columns = to_drop)

del df, train, test, Fraud
# y_train = train['isFraud'].copy()



# # Drop target

# X_train = train.drop('isFraud', axis=1)

# X_test = test.copy()
print(X_train.shape)

print(X_test.shape)
X_train.head()
# df = pd.concat([X_train_full, y_train_full], axis=1)

# df.head()
# df.describe()
# # sample dataframe to obtain better exectution time

# N = 5000

# df = df.sample(n = N, random_state = 0)

# sns.pairplot(data=df)
# plt.figure(figsize=(10,10))

# sns.boxplot(x="features", y="value", hue="isFraud", data=df)

# plt.xticks(rotation=90)
# del df
# # Label Encoding

# for f in X_train_full.columns:

#     if X_train_full[f].dtype=='object' or X_test[f].dtype=='object': 

#         lbl = preprocessing.LabelEncoder()

#         lbl.fit(list(X_train_full[f].values) + list(X_test[f].values))

#         X_train_full[f] = lbl.transform(list(X_train_full[f].values))

#         X_test[f] = lbl.transform(list(X_test[f].values)) 
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state=0)
## REducing memory

X_train = reduce_mem_usage(X_train)

X_val = reduce_mem_usage(X_val)
X_train.shape
m = RandomForestClassifier(n_jobs=-1, n_estimators = 200)

m.fit(X_train, y_train)



print(roc_auc_score(y_val,m.predict_proba(X_val)[:,1] ))
sample_submission['isFraud'] = m.predict_proba(X_test)[:,1]

sample_submission.to_csv('base_RF.csv')
# top N importances

N = 10

importances = m.feature_importances_

std = np.std([tree.feature_importances_ for tree in m.estimators_],

             axis=0)



# create a dataframe

importances_df = pd.DataFrame({'variable':X_train.columns, 'importance': importances})



top_N = importances_df.sort_values(by=['importance'], ascending=False).head(N)



top_N
sns.barplot(data = top_N, y = "variable", x = "importance", color = 'steelblue')
print(X_train.shape)

print(X_val.shape[0])
from sklearn.model_selection import ParameterGrid



# Create a dictionary of hyperparameters to search

grid = {'n_estimators':[150,200,250], 'max_depth': [10,15,20], 'max_features': [10,15, 20], 'random_state': [0]}

test_scores = []



# Loop through the parameter grid, set the hyperparameters, and save the scores

for g in ParameterGrid(grid):

    m.set_params(**g)  # ** is "unpacking" the dictionary

    m.fit(X_train, y_train)

    #test_scores.append(m.score(X_val, y_val))

    test_scores.append(roc_auc_score(y_val, m.predict_proba(X_val)[:,1]))



# Find best hyperparameters from the test score and print

best_idx = np.argmax(test_scores)

print(test_scores[best_idx], ParameterGrid(grid)[best_idx])
# pick the best results

m.set_params(**ParameterGrid(grid)[best_idx])
print(roc_auc_score(y_val,m.predict_proba(X_val)[:,1] ))
sample_submission['isFraud'] = m.predict_proba(X_test)[:,1]

sample_submission.to_csv('tuned_RF.csv')