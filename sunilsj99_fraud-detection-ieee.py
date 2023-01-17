import os

import time

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

from tqdm import tqdm

import matplotlib.pyplot as plt

from catboost import CatBoostClassifier

from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_auc_score

warnings.filterwarnings('ignore')

print(os.listdir("../input"))



%matplotlib inline
%%time

train_transaction = pd.read_csv('../input/train_transaction.csv')

train_identity = pd.read_csv('../input/train_identity.csv')

test_transaction = pd.read_csv('../input/test_transaction.csv')

test_identity = pd.read_csv('../input/test_identity.csv')
%%time

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
del train_transaction

del train_identity

del test_transaction

del test_identity
def reduce_mem_usage(props):

    start_mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in props.columns:

        if props[col].dtype != object:  # Exclude strings

            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",props[col].dtype)

            

            # make variables for Int, max and min

            IsInt = False

            mx = props[col].max()

            mn = props[col].min()

            

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(props[col]).all(): 

                NAlist.append(col)

                props[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = props[col].fillna(0).astype(np.int64)

            result = (props[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True



            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        props[col] = props[col].astype(np.uint8)

                    elif mx < 65535:

                        props[col] = props[col].astype(np.uint16)

                    elif mx < 4294967295:

                        props[col] = props[col].astype(np.uint32)

                    else:

                        props[col] = props[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        props[col] = props[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        props[col] = props[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        props[col] = props[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        props[col] = props[col].astype(np.int64)    

            

            # Make float datatypes 32 bit

            else:

                props[col] = props[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",props[col].dtype)

            print("******************************")

    

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return props
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
train.shape
test.shape
train.columns
train.head()
null_percent = train.isnull().sum()/train.shape[0]*100



cols_to_drop = np.array(null_percent[null_percent > 50].index)



cols_to_drop
train = train.drop(cols_to_drop, axis=1)

test = test.drop(cols_to_drop,axis=1)
null_percent = test.isnull().sum()/train.shape[0]*100

null_percent[null_percent > 0]
null_cols = ['card4', 'card6', 'P_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M6']



for i in null_cols:

    print('data type of {} is {}'.format(i, str(train[i].dtype)))

    train[i] = train[i].replace(np.nan, train[i].mode()[0])

    test[i] = test[i].replace(np.nan, train[i].mode()[0])

    print('Filled the null values of column {}'.format(i))

    print('--------------------------------------------')
X = train.drop('isFraud', axis=1)

y = train['isFraud']
cat_data = X.select_dtypes(include='object')

num_data = X.select_dtypes(exclude='object')



cat_cols = cat_data.columns.values

num_cols = num_data.columns.values



print('Categorical Columns : ',cat_cols)

print('Numerical Columns : ',num_cols)
fig = plt.figure(figsize=(20,15))



j = 1

for i in cat_cols:

    if(i == 'P_emaildomain'):

        continue

    plt.subplot(3,3,j)

    sns.countplot(x=X[i], palette='winter_r')

    j = j + 1

    

plt.show()
plt.figure(figsize=(20,8))

sns.countplot(x=X['P_emaildomain'], color='blue')

plt.xticks(rotation = 90)

plt.show()
fig = plt.figure(figsize=(20,15))



j = 1

for i in num_cols[1:10]:

    plt.subplot(3,3,j)

    sns.distplot(a=X[i])

    j = j + 1

    

plt.show()
fig = plt.figure(figsize=(20,15))



j = 1

for i in num_cols[-23:-11]:

    plt.subplot(3,4,j)

    sns.distplot(a=X[i])

    j = j + 1

    

plt.show()
sns.countplot(x=y, palette='gist_rainbow')

plt.title('Fraud or Not')

plt.show()
df1 = train[train['isFraud'] == 0]

not_fraud = df1['TransactionAmt'].apply(np.log) #we will apply log transformation to get better visualization 



df2 = train[train['isFraud'] == 1]

fraud = df2['TransactionAmt'].apply(np.log) #we will apply log transformation to get better visualization 



plt.figure(figsize=(20, 7))



sns.distplot(a=not_fraud, label='Not Fraud')

sns.distplot(a=fraud, label='Fraud')



plt.legend()
X['TransactionAmt'] = X['TransactionAmt'].apply(np.log)

test['TransactionAmt'] = test['TransactionAmt'].apply(np.log)
X = X.drop('TransactionDT', axis=1)

test = test.drop('TransactionDT', axis=1)
del train
from sklearn.preprocessing import LabelEncoder



for i in tqdm(cat_cols): 

    label = LabelEncoder()

    label.fit(list(X[i].values)+list(test[i].values))

    X[i] = label.transform(list(X[i].values))

    test[i] = label.transform(list(test[i].values))
X.head()
X = X.drop('TransactionID', axis=1)

test = test.drop('TransactionID', axis=1)
c = X.corr()
plt.figure(figsize=(20,20))

sns.heatmap(c)
col_corr = set()

for i in range(len(c.columns)):

    for j in range(i):

        if (c.iloc[i, j] >= 0.95) and (c.columns[j] not in col_corr):

            colname = c.columns[i] # getting the name of column

            col_corr.add(colname)
cols = X.columns

print('{} and {}'.format(len(cols), len(col_corr)))
final_columns = []



for i in cols:

    if i in col_corr:

        continue

    else:

        final_columns.append(i)
X1 = X[final_columns]

test1 = test[final_columns]
print(X1.shape)

print(test1.shape)
plt.figure(figsize=(20,20))

sns.heatmap(X1.corr())
del X

del test
params = {'objective': 'binary',  

          'learning_rate': 0.1, 

          'num_leaves': 256,

          'is_unbalance': True, 

          'metric': 'auc', 

          'feature_fraction': 0.8, 

          'verbosity': -1,

          'random_state': 42

          }
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb



kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_score_lgb = []

i = 1

#predictions = np.zeros(test1.shape[0])



print('5 Fold Stratified Cross Validation')

print('-----------------------------------')

for train_index, test_index in kf.split(X1, y):

    print('Fold no. {}'.format(i))

    xtr, ytr = X1.loc[train_index], y.loc[train_index]

    xv, yv = X1.loc[test_index], y.loc[test_index]

    

    df_train = lgb.Dataset(xtr, label=ytr)

    df_val = lgb.Dataset(xv, label=yv)

    

    clf1 = lgb.train(params, num_boost_round = 5000,train_set = df_train, valid_sets=[df_train, df_val], verbose_eval=400, early_stopping_rounds=200)

    ypred =  clf1.predict(xv)

    score = f1_score(yv, ypred.round())

    print('F1-Score : {}'.format(score))

    cv_score_lgb.append(score)

    #predictions = predictions + clf1.predict(test1)/5

    i += 1

    print('-------------------------------------')
print('Mean AUC Score : {}'.format(np.array(cv_score_lgb).mean()))
df_train = lgb.Dataset(X1, label=y)
clf_final = lgb.train(params, num_boost_round = 1200,train_set = df_train, valid_sets=[df_train],

                 verbose_eval=400, early_stopping_rounds=200)
# a = pd.Series(y).value_counts()

# a[1]/len(y)*100
# from sklearn.model_selection import StratifiedKFold



# kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# cv_score = []

# i = 1

# predictions = np.zeros(test1.shape[0])

# print('5 Fold Stratified Cross Validation')

# print('-----------------------------------')

# for train_index, test_index in kf.split(X1, y):

#     print('Fold no. {}'.format(i))

#     xtr, ytr = X1.loc[train_index], y.loc[train_index]

#     xv, yv = X1.loc[test_index], y.loc[test_index]

    

#     clf = CatBoostClassifier(task_type='GPU', eval_metric='AUC', loss_function='Logloss', use_best_model=True,

#                           silent=True, class_weights= [0.01, 0.99],

#                          random_state=42, iterations=5000, od_type='Iter', od_wait=200, grow_policy='Lossguide',

#                         max_depth = 7, l2_leaf_reg= 0.5)

#     clf.fit(xtr, ytr, eval_set=(xv, yv))

#     score = roc_auc_score(yv, clf.predict(xv))

#     ypreds = clf.predict_proba(test1)/5

#     predictions += ypreds[:,1]

#     print('AUC score Train : {} \t AUC score Val : {}'.format(roc_auc_score(ytr, clf.predict(xtr)), score))

#     cv_score.append(score)

#     i += 1

#     print('-------------------------------------')
# print('Mean AUC Score : {}'.format(np.array(cv_score).mean()))
# clf = CatBoostClassifier(task_type='GPU', eval_metric='AUC', loss_function='Logloss',

#                          class_weights=[0.1, 0.9],

#                           random_state=42, iterations=5000, od_type='Iter', od_wait=200, grow_policy='Lossguide', max_depth=8)

# clf.fit(X1, y)
probs = clf_final.predict(test1)
sub = pd.read_csv('../input/sample_submission.csv')
sub['isFraud'] = probs
sub.head()
sub.to_csv('submission.csv', index=False)
feature_dict = {'Features': clf_final.feature_name(), 'Importance': clf_final.feature_importance()}
feature_imp = pd.DataFrame(feature_dict).sort_values(by=['Importance'], ascending=False)
feature_imp.head(10)
plt.figure(figsize=(10,7))

df_imp = feature_imp.head(10)

sns.barplot(y=df_imp['Features'], x=df_imp['Importance'], palette='winter_r')