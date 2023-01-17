import pandas as pd
import numpy as np
import time
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns


import os
print(os.listdir("../input")) # Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
submit_df = pd.read_csv('../input/sample_submission.csv')
print(train_df.shape)
train_df.head()
print(test_df.shape)
test_df.head()
# it'll be useful to combine the train and test data into a single dataframe to see how they differ

df = pd.concat([train_df, test_df], sort=False)
df['is_train'] = df['target'].notnull() 
df['first_active_month'] = df['first_active_month'].apply(pd.to_datetime)
df.sample(4)
# card ids are indeed unique identifiers
print(df.card_id.nunique() / df.shape[0])
# lets look at the first_active_month column. I suspect that train and test differ greatly
def df_pivot_distribution(df, index, split='is_train', count_on='card_id'):
    return df.groupby([index, split]).count()[count_on].reset_index().pivot(index=index, columns=split, values=count_on)#.plot.bar(figsize=(20,6), stacked=True)

df_pivot_distribution(df, 'first_active_month').plot.bar(stacked=True, figsize=(15, 4))
df_pivot_distribution(df, 'feature_1').plot.bar()
df_pivot_distribution(df, 'feature_2').plot.bar()
df_pivot_distribution(df, 'feature_3').plot.bar()
df['target'].describe()
# since the columns labeled 'features' are already engineered, let's try to get useful information from the date field

df['first_active_mo'] = df['first_active_month'].dt.month
df['first_active_yr'] = df['first_active_month'].dt.year
# Our 'feature' features appear to be categorical - rather than actually numeric. We may get clearer signal if we
# don't treat them as sequential

cat_columns = ['feature_1', 'feature_2', 'feature_3', 'first_active_yr']
df = pd.get_dummies(df, columns=cat_columns)
df.head()
# params taken from public kernel https://www.kaggle.com/peterhurford/you-re-going-to-want-more-categories-lb-3-737
param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.0041,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1}

# seperate our train, test, and targets
train_df = df.loc[df['is_train'] == True].set_index('card_id')
test_df = df.loc[df['is_train'] == False].set_index('card_id')
target = train_df['target']

# we only want to pass our import numeric features to the model
ignored_columns = ['first_active_month', 'target', 'is_train']
train_df = train_df[[ix for ix in train_df.columns if ix not in ignored_columns]]
test_df = test_df[[ix for ix in test_df.columns if ix not in ignored_columns]]

#initalize our outputs and our folds
folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
start = time.time()
features = list(train_df.columns)
feature_importance_df = pd.DataFrame()

# loop through the folds, train the model, and make a prediction
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    trn_data = lgb.Dataset(train_df.iloc[trn_idx].values, label=target.iloc[trn_idx].values)
    val_data = lgb.Dataset(train_df.iloc[val_idx].values, label=target.iloc[val_idx].values)
    
    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx].values, num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test_df.values, num_iteration=clf.best_iteration) / folds.n_splits
    
print("Total OOF RMSE: {}".format(np.sqrt(mean_squared_error(target, oof))))
print("Our Baseline model (predicting all 0's) RMSE:{}".format(np.sqrt(mean_squared_error(target, np.array([0] * len(oof))))) )
test_df['target'] = predictions
test_df.reset_index()[['card_id', 'target']].to_csv('submission.csv', index=False)