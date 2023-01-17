import numpy as np

import pandas as pd

from sklearn import preprocessing, metrics

import lightgbm as lgb

import pickle

import gc



import os

print(pd.__version__)



dirpath = os.getcwd()

print("current directory is : " + dirpath)

print(os.listdir("/kaggle/input"))

PATH = "../input/cat-in-the-dat/"



train_raw = pd.read_csv(PATH + 'train.csv')

test_raw = pd.read_csv(PATH + 'test.csv')



print(train_raw.shape)

print(test_raw.shape)

train_raw.head(3)

print(train_raw.axes)

print(train_raw.dtypes)



def get_data_splits(dataframe, valid_fraction=0.1):



    #dataframe = dataframe.sort_values('click_time')  #uncomment if you have time-series data

    valid_rows = int(len(dataframe) * valid_fraction)

    train = dataframe[:-valid_rows * 2]

    # valid size == test size, last two sections of the data

    valid = dataframe[-valid_rows * 2:-valid_rows]

    test = dataframe[-valid_rows:]

    

    return train, valid, test



def train_model(train, valid, test=None, feature_cols=None):

    if feature_cols is None:

        feature_cols = train.columns.drop(['id', 'target'])

    dtrain = lgb.Dataset(train[feature_cols], label=train['target'])

    dvalid = lgb.Dataset(valid[feature_cols], label=valid['target'])

    

    param = {'num_leaves': 64, 'objective': 'binary', 

             'metric': 'auc', 'seed': 7}

    num_round = 1000

    print("Training model!")

    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], 

                    early_stopping_rounds=20, verbose_eval=False)

    

    valid_pred = bst.predict(valid[feature_cols])

    valid_score = metrics.roc_auc_score(valid['target'], valid_pred)

    print(f"Validation AUC score: {valid_score}")

    

    if test is not None: 

        test_pred = bst.predict(test[feature_cols])

        test_score = metrics.roc_auc_score(test['target'], test_pred)

        return bst, valid_score, test_score

    else:

        return bst, valid_score
## replacing extra levels in test with mode:

test_all_feat = test_raw.copy()



for iter in ('nom_8','nom_9'):

    s = pd.Series(list(set(test_all_feat[iter].values) - set(train_raw[iter].values)))

    s1 = pd.Series(test_all_feat[iter].isin(s))

    test_all_feat.loc[s1,iter] = test_all_feat[iter].mode().values

for iter in ('nom_8','nom_9'):

    s = pd.Series(list(set(test_all_feat[iter].values) - set(train_raw[iter].values)))

    s1 = pd.Series(test_all_feat[iter].isin(s))

    print(s1.value_counts())

train_bench = train_raw.copy()

train_bench = train_bench.assign(bin_3 = (train_bench["bin_3"]=="T").astype(int))

train_bench = train_bench.assign(bin_4 = (train_bench["bin_4"]=="Y").astype(int))

print(train_bench.iloc[:4,:6])

      

test_all_feat = test_all_feat.assign(bin_3 = (test_all_feat["bin_3"]=="T").astype(int))

test_all_feat = test_all_feat.assign(bin_4 = (test_all_feat["bin_4"]=="Y").astype(int))

print(test_all_feat.iloc[:4,:6])



cat_features =list(train_bench.columns[6:22])

print(cat_features)



from sklearn.preprocessing import LabelEncoder



for iter in cat_features:

    encoder = LabelEncoder()

    encoder.fit(train_bench[iter])

    train_bench[iter] = encoder.transform(train_bench[iter])

    test_all_feat[iter] = encoder.transform(test_all_feat[iter])

    

test_all_feat.head(2)



train_all_feat = train_bench.copy()
train, valid, test = get_data_splits(train_bench)

for each in [train, valid, test]:

    print(f"Target fraction = {each.target.mean():.4f}")

    

_, baseline_score, test_score = train_model(train, valid, test)



# peeking at test score just once. we will re-check it after we are done with Feature enginering.

print(f"Test AUC score: {test_score}")
cat_features = list(train_bench.columns[6:24])

print(cat_features)

bin_features = list(train_bench.columns[1:6])

print(bin_features)



import category_encoders as ce

count_enc = ce.CountEncoder()

count_encoded = count_enc.fit_transform(train_bench[cat_features])



data = train_bench.join(count_encoded.add_suffix("_count"))

print(data.shape)

data.head(3)



data_all_feat = data.copy() ## collecting all features for subsequent feature selection



# Training a model on the baseline data

train, valid, test = get_data_splits(data)

print("Score with count encoding")

bst = train_model(train, valid)

train_bench.head(3)

import category_encoders as ce

cat_features = list(train_bench.columns[6:24])

print(cat_features)



# Create the encoder itself

target_enc = ce.TargetEncoder(cols=cat_features)



train, valid, _ = get_data_splits(train_bench)



# Fit the encoder using the categorical features and target

target_enc.fit(train[cat_features], train['target'])



# Transform the features, rename the columns with _target suffix, and join to dataframe

train = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))

valid = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))



print(train.shape)

print("Score with target encoding")

bst = train_model(train, valid)
target_enc = ce.CatBoostEncoder(cols=cat_features)



train, valid, _ = get_data_splits(train_bench)

target_enc.fit(train[cat_features], train['target'])



train = train.join(target_enc.transform(train[cat_features]).add_suffix('_cb'))

valid = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_cb'))



print(train.shape)

print("Score with CatBoost target encoding")

bst = train_model(train, valid)
data_all_feat = data_all_feat.join(target_enc.transform(train_bench[cat_features]).add_suffix('_cb'))
import itertools



cat_features = list(train_bench.columns[1:24])

print(cat_features)



interactions = pd.DataFrame(index=train_bench.index)

list(itertools.combinations(cat_features,2))[0][1]

for iter in list(itertools.combinations(cat_features,2)):

    naming = (iter[0]+"_"+iter[1])

    interactions = interactions.assign(**{naming : (train_bench[iter[0]].astype(str) + "_" + train_bench[iter[1]].astype(str))})

    encoder = preprocessing.LabelEncoder()

    interactions[naming] = encoder.fit_transform(interactions[naming])

interactions.head()
data = train_bench.join(interactions)

print(data.shape)



print("Score with interactions")

train, valid, test = get_data_splits(data)

_ = train_model(train, valid)



del data
data_all_feat = data_all_feat.join(interactions)

data_all_feat.shape
from sklearn.feature_selection import SelectKBest, f_classif

feature_cols = data_all_feat.columns.drop(['target'])

train, valid, test = get_data_splits(data_all_feat)

print(train.shape)



selector = SelectKBest(f_classif, k=110)  # Create the selector, keeping 110 features, k should be optimized further



# Use the selector to retrieve the best features

X_new = selector.fit_transform(train[feature_cols],train['target'])



# Get back the kept features as a DataFrame with dropped columns as all 0s

selected_features = pd.DataFrame(selector.inverse_transform(X_new), index=train.index, columns=feature_cols)

#selected_features.head()



# Find the columns that were dropped

dropped_columns = selected_features.columns[selected_features.var(axis=0)==0]

print(len(dropped_columns))

print(dropped_columns)

dropped_columns = dropped_columns.drop('id')

dropped_columns_KBest = dropped_columns.copy()  ##saving for final model



_, baseline_score, test_score = train_model(train.drop(dropped_columns, axis=1), valid.drop(dropped_columns, axis=1),test.drop(dropped_columns, axis=1))

print(f"Test AUC score after LogReg_L1 feature selection: {test_score}")



del X_new
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel



def select_features_l1(X, y):

    """ Return selected features using logistic regression with an L1 penalty """

    logistic = LogisticRegression(C=0.03, random_state=7, penalty="l1")

    logistic_fit = logistic.fit(X,y)

    

    selector = SelectFromModel(logistic_fit, prefit=True)

    X_new = selector.transform(X)



    # Get back the features we've kept, zero out all other features

    selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                    index=X.index, 

                                    columns=X.columns)



    # Dropped columns have values of all 0s, so var is 0, drop them

    selected_columns = selected_features.columns[selected_features.var() != 0]

    

    return selected_columns
feature_cols = data_all_feat.columns.drop(['target'])

train, valid, test = get_data_splits(data_all_feat, valid_fraction=0.1)  #####  use smaller train subset (e.g. valid_fraction=0.3 ) to speed up L1 feature selection

print(train.shape)



feature_cols = data_all_feat.columns.drop(['target'])

selected = select_features_l1(train[feature_cols], train['target'])

#print(selected)

dropped_columns = feature_cols.drop(selected)

dropped_columns = dropped_columns.drop('id')

print(len(dropped_columns))

print(dropped_columns)

dropped_columns_L1 = dropped_columns.copy()  ##saving for final model



train, valid, test = get_data_splits(data_all_feat, valid_fraction=0.1)



_, baseline_score, test_score = train_model(train.drop(dropped_columns, axis=1), valid.drop(dropped_columns, axis=1),test.drop(dropped_columns, axis=1))

print(f"Test AUC score after LogReg_L1 feature selection: {test_score}")
import category_encoders as ce

cat_features = list(train_bench.columns[6:24])

#print(cat_features)

bin_features = list(train_bench.columns[1:6])

#print(bin_features)



train_encoded = train_bench[cat_features].copy()

test_encoded = test_all_feat[cat_features].copy()



enc = ce.CountEncoder(cols=cat_features).fit(train_bench[cat_features])

train_encoded = enc.transform(train_bench[cat_features])

test_encoded = enc.transform(test_all_feat[cat_features])



print(train_encoded.shape)

print(test_encoded.shape)



train_all_feat = train_all_feat.join(train_encoded.add_suffix("_count"))

test_all_feat = test_all_feat.join(test_encoded.add_suffix("_count"))



print(train_all_feat.shape)



del train_encoded

del test_encoded
target_enc = ce.CatBoostEncoder(cols=cat_features)



target_enc.fit(train_bench[cat_features], train_bench['target'])



train_all_feat = train_all_feat.join(target_enc.transform(train_bench[cat_features]).add_suffix('_cb'))

test_all_feat = test_all_feat.join(target_enc.transform(test_all_feat[cat_features]).add_suffix('_cb'))



print(train_all_feat.shape)

print(test_all_feat.shape)
import itertools



cat_features = list(train_bench.columns[1:24])

print(cat_features)



interactions = pd.DataFrame(index=train_bench.index)

interactions_test = pd.DataFrame(index=test_all_feat.index)



list(itertools.combinations(cat_features,2))[0][1]

for iter in list(itertools.combinations(cat_features,2)):

    naming = (iter[0]+"_"+iter[1])

    interactions = interactions.assign(**{naming : (train_bench[iter[0]].astype(str) + "_" + train_bench[iter[1]].astype(str))})

    interactions_test = interactions_test.assign(**{naming : (test_all_feat[iter[0]].astype(str) + "_" + test_all_feat[iter[1]].astype(str))})

    

print(interactions.shape)

print(interactions_test.shape)



gc.collect()



del data_all_feat

gc.collect()



data = pd.concat([interactions,interactions_test],axis=0, ignore_index = True)



for iter in list(interactions.columns):

    encoder = LabelEncoder()

    encoder.fit(data[iter])

    data[iter] = encoder.transform(data[iter])



interactions = data.iloc[0:len(interactions)]

interactions_test = data.iloc[len(interactions):len(data)]

interactions_test.reset_index(drop=True,inplace = True)



print(interactions.shape)

print(interactions_test.shape)
train_all_feat = train_all_feat.join(interactions)

test_all_feat = test_all_feat.join(interactions_test)



print(train_all_feat.shape)

print(test_all_feat.shape)



del interactions, interactions_test

gc.collect()
dropped_columns = dropped_columns_KBest

#print(dropped_columns)



feature_cols = train_all_feat.drop(dropped_columns, axis=1).columns.drop(['id', 'target'])

print(feature_cols)



valid_fraction = 0.1

valid_rows = int(len(train_all_feat) * valid_fraction)

train = train_all_feat[:-valid_rows]

valid = train_all_feat[-valid_rows:]

print(train.shape)

print(valid.shape)



dtrain = lgb.Dataset(train[feature_cols], label=train['target'])

dvalid = lgb.Dataset(valid[feature_cols], label=valid['target'])

    

param = {'num_leaves': 64, 'objective': 'binary', 'metric': 'auc', 'seed': 77}

num_round = 1000

print("Training model!")

bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=20, verbose_eval= 50)



num_iteration=bst.best_iteration

print(num_iteration)

    

valid_pred = bst.predict(valid[feature_cols])

valid_score = metrics.roc_auc_score(valid['target'], valid_pred)

print(f"Validation AUC score: {valid_score}")

    

## training on full train set

dtrain = lgb.Dataset(train_all_feat[feature_cols], label=train_all_feat['target'])

param = {'num_leaves': 64, 'objective': 'binary', 'metric': 'auc', 'seed': 77}

num_round = num_iteration

print("Training model!")

bst = lgb.train(param, dtrain, num_round,  verbose_eval=True)



test_pred = bst.predict(test_all_feat[feature_cols])



sub1 = pd.read_csv(PATH + 'sample_submission.csv')

sub1.head()

sub1['target'] =test_pred

sub1.to_csv("submission_KBest.csv",index=False)
dropped_columns = dropped_columns_L1

#print(dropped_columns)



feature_cols = train_all_feat.drop(dropped_columns, axis=1).columns.drop(['id', 'target'])

print(feature_cols)



valid_fraction = 0.1

valid_rows = int(len(train_all_feat) * valid_fraction)

train = train_all_feat[:-valid_rows]

valid = train_all_feat[-valid_rows:]

print(train.shape)

print(valid.shape)



dtrain = lgb.Dataset(train[feature_cols], label=train['target'])

dvalid = lgb.Dataset(valid[feature_cols], label=valid['target'])

    

param = {'num_leaves': 64, 'objective': 'binary', 'metric': 'auc', 'seed': 77}

num_round = 1000

print("Training model!")

bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=20, verbose_eval=50)



num_iteration=bst.best_iteration

print(num_iteration)



valid_pred = bst.predict(valid[feature_cols])

valid_score = metrics.roc_auc_score(valid['target'], valid_pred)

print(f"Validation AUC score: {valid_score}")

    

## training on full train set

dtrain = lgb.Dataset(train_all_feat[feature_cols], label=train_all_feat['target'])

param = {'num_leaves': 64, 'objective': 'binary', 'metric': 'auc', 'seed': 77}

num_round = num_iteration

print("Training model!")

bst = lgb.train(param, dtrain, num_round,  verbose_eval=True)



test_pred = bst.predict(test_all_feat[feature_cols])



sub2 = pd.read_csv(PATH + 'sample_submission.csv')

sub2.head()

sub2['target'] =test_pred

sub2.to_csv("submission_L1.csv",index=False)
sub3 = pd.read_csv(PATH + 'sample_submission.csv')

sub3['target'] = (sub1['target']+sub2['target'])/2

sub3.to_csv("submission_1_2.csv",index=False)
feature_cols = train_all_feat.columns.drop(['id', 'target'])



valid_fraction = 0.1

valid_rows = int(len(train_all_feat) * valid_fraction)

train = train_all_feat[:-valid_rows]

valid = train_all_feat[-valid_rows:]

print(train.shape)

print(valid.shape)



dtrain = lgb.Dataset(train[feature_cols], label=train['target'])

dvalid = lgb.Dataset(valid[feature_cols], label=valid['target'])

    

param = {'num_leaves': 64, 'objective': 'binary', 'metric': 'auc', 'seed': 77}

num_round = 1000

print("Training model!")

bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=20, verbose_eval=50)



num_iteration=bst.best_iteration

print(num_iteration)



valid_pred = bst.predict(valid[feature_cols])

valid_score = metrics.roc_auc_score(valid['target'], valid_pred)

print(f"Validation AUC score: {valid_score}")

    

## training on full train set

dtrain = lgb.Dataset(train_all_feat[feature_cols], label=train_all_feat['target'])

param = {'num_leaves': 64, 'objective': 'binary', 'metric': 'auc', 'seed': 77}

num_round = num_iteration

print("Training model!")

bst = lgb.train(param, dtrain, num_round,  verbose_eval=True)



test_pred = bst.predict(test_all_feat[feature_cols])



sub4 = pd.read_csv(PATH + 'sample_submission.csv')

sub4.head()

sub4['target'] =test_pred

sub4.to_csv("submission_full_set.csv",index=False)
train_all_feat.to_csv("train_all_feat.csv.zip", index = False, compression = 'zip')

test_all_feat.to_csv("test_all_feat.csv.zip", index = False, compression = 'zip')



with open('dropped_columns_L1.data', 'wb') as filehandle:

    pickle.dump(dropped_columns_L1, filehandle)

with open('dropped_columns_KBest.data', 'wb') as filehandle:

    pickle.dump(dropped_columns_KBest, filehandle)