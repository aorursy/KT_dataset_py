import pandas as pd
import numpy as np
import datetime
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import lightgbm as lgb
import matplotlib.pyplot as plt
%matplotlib inline
#Import data
print('Importing data...')
df_train = pd.read_csv('../input/train.csv')
df_history = pd.read_csv("../input/historical_transactions.csv")
#Preprocess transactions
print('Preprocessing historical transactions...')
df_history['authorized_flag'] = df_history['authorized_flag'].map({'Y':1, 'N':0})
df_history['category_1'] = df_history['category_1'].map({'Y':1, 'N':0})
df_history['purchase_date'] = pd.to_datetime(df_history['purchase_date'])
last_date_hist = datetime.datetime(2018, 2, 28)
df_history['time_since_purchase_date'] = ((last_date_hist - df_history['purchase_date']).dt.days)
df_history.loc[:, 'purchase_date'] = pd.DatetimeIndex(df_history['purchase_date']).\
                                      astype(np.int64) * 1e-9

df_history['installments'] = df_history['installments'].replace(999,-1)
cols_with_nulls = ['city_id', 'state_id', 'subsector_id', 'installments']
for col in cols_with_nulls:
    df_history[col] = df_history[col].replace(-1, np.nan)

#Perform aggregations by card ID
print('Aggregating historical transactions...')

agg_func = {
        'authorized_flag': ['mean'],
        'city_id': ['nunique'], 
        'category_1': ['sum', 'mean'],
        'installments': ['median', 'max'],
        'category_3': ['nunique'],
        'merchant_category_id': ['nunique'], 
        'merchant_id': ['nunique'],
        'month_lag': ['min', 'max'],
        'purchase_amount': ['sum', 'median', 'max', 'min'],
        'purchase_date': ['min', 'max'],
        'time_since_purchase_date': ['min', 'max', 'mean'],
        'category_2': ['nunique'], 
        'state_id': ['nunique'], 
        'subsector_id': ['nunique']
        }


agg_history = df_history.groupby(['card_id']).agg(agg_func)
agg_history.columns = ['hist_' + '_'.join(col).strip() for col in agg_history.columns.values]
agg_history.reset_index(inplace=True)
#Merge with train and test
print('Merging all data...')
df_train_all = pd.merge(df_train, agg_history, on='card_id', how='left')
#Split initial train set into new train and test sets
y_label_regr = df_train_all['target']

df_train_all = df_train_all.drop(['target',
                                    'first_active_month', 
                                    'card_id'
                                    ],
                                     axis = 1)

train_x, test_x, train_y, test_y = train_test_split(df_train_all, y_label_regr, test_size=0.7, random_state=42)

train_x.reset_index(inplace=True, drop = True)
test_x.reset_index(inplace=True, drop = True)
train_y.reset_index(inplace=True, drop = True)
test_y.reset_index(inplace=True, drop = True)
#Train LightGBM model on original data
param = {'num_leaves': 111,
         'min_data_in_leaf': 149,
         'objective':'regression',
         'max_depth': 9,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.7522,
         "bagging_freq": 1,
         "bagging_fraction": 0.7083 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.3134,
         "random_state": 133,
         "verbosity": -1}

features = train_x.columns
folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof_normal = np.zeros(len(df_train_all))
predictions_normal = np.zeros(len(test_x))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x.values, train_y)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train_x.iloc[trn_idx][features],
                           label=train_y[trn_idx],
                           #categorical_feature=cat_feats
                           )
    val_data = lgb.Dataset(train_x.iloc[val_idx][features],
                           label=train_y[val_idx],
                           #categorical_feature=cat_feats
                           )

    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds=200)

    oof_normal[val_idx] = clf.predict(train_x.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions_normal += clf.predict(test_x[features], num_iteration=clf.best_iteration) / folds.n_splits

#Perform Principal Component Analysis  
pca = PCA()
pca.fit(train_x)
pca.transform(train_x)
#Visualize Scree plot
fig,ax=plt.subplots(1,2,figsize=(12,6))
pc_total=np.arange(1,pca.n_components_+1)
ax[0].plot(pc_total,np.cumsum(pca.explained_variance_ratio_))
ax[0].set_xticks(pc_total)
ax[0].set_xlabel('Principal Components')
ax[0].set_ylabel('Cumulative explained variance')
###############################################################
ax[1].plot(pc_total,pca.explained_variance_)
ax[1].set_xticks(pc_total)
ax[1].set_xlabel('Principal Components')
ax[1].set_ylabel('Explained Variance Ratio')
fig.suptitle('SCREE PLOT')
plt.show()
var_exp_3 = sum(pca.explained_variance_ratio_[:3])
print('Variance explained by the first 3 PCA components:', var_exp_3)
#Visualize Biplot
#Unfortunately this throws a size error here but not in my local notebook. I am just leaving the 
#code here commented out so you can use it on your own

# y=pca.fit_transform(train_x)

# plt.figure(figsize = (12,10))

# xvector = pca.components_[0] 
# yvector = pca.components_[1]

# xs = y[:,0]
# ys = y[:,1]

# ## visualize projections
# for i in range(len(xvector)):
#     plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
#               color='darkred', width=0.2, head_width=0.5)
#     plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,
#              list(train_x.columns)[i], color='darkred', fontsize=6)

# plt.scatter(xs,ys,c='b',alpha=0.02)
# plt.axhline(0, color='black',alpha=0.8)
# plt.axvline(0, color='black',alpha=0.8)
# plt.xlim(-300,300)
# plt.ylim(-300,300)
# plt.show()
#Create train and test sets from the first 3 PCA components
pca_train_x= pca.transform(train_x)
pca_train_x = pca_train_x[:,:3]
pca_train_x = pd.DataFrame(pca_train_x, columns=['comp1', 'comp2', 'comp3'])

pca_test_x= pca.transform(test_x)
pca_test_x = pca_test_x[:,:3]
pca_test_x = pd.DataFrame(pca_test_x, columns=['comp1', 'comp2', 'comp3'])
#Train LightGBM model on these 3 PCA components only
param = {#'num_leaves': 21,
         #'min_data_in_leaf': 49,
         'objective':'regression',
         'max_depth': 8,
         'learning_rate': 0.001,
         #"boosting": "gbdt",
         #"feature_fraction": 0.5,
         #"bagging_freq": 1,
         #"bagging_fraction": 0.5 ,
         #"bagging_seed": 11,
         "metric": 'rmse',
         #"lambda_l1": 0.3134,
         "random_state": 133,
         #"is_unbalance": True,
         "verbosity": -1}

features = pca_train_x.columns
folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof_pca = np.zeros(len(pca_train_x))
predictions_pca = np.zeros(len(pca_test_x))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(pca_train_x.values, train_y)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(pca_train_x.iloc[trn_idx][features],
                           label=train_y[trn_idx],
                           #categorical_feature=cat_feats
                           )
    val_data = lgb.Dataset(pca_train_x.iloc[val_idx][features],
                           label=train_y[val_idx],
                           #categorical_feature=cat_feats
                           )

    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds=200)

    oof_pca[val_idx] = clf.predict(pca_train_x.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions_pca += clf.predict(pca_test_x[features], num_iteration=clf.best_iteration) / folds.n_splits

#Compare performance
print("RMSE test normal: {:<8.5f}".format(mean_squared_error(predictions_normal, test_y) ** 0.5))
print("RMSE test PCA: {:<8.5f}".format(mean_squared_error(predictions_pca, test_y) ** 0.5))
#Add these 3 PCA components as features on original data and re-run model
train_x_2 = pd.concat([train_x, pca_train_x], axis = 1)
test_x_2 = pd.concat([test_x, pca_test_x], axis = 1)

del train_x
del test_x
del pca_train_x
del pca_test_x

#Train LightGBM model
param = {'num_leaves': 111,
         'min_data_in_leaf': 149,
         'objective':'regression',
         'max_depth': 9,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.7522,
         "bagging_freq": 1,
         "bagging_fraction": 0.7083 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.3134,
         "random_state": 133,
         "verbosity": -1}

features = train_x_2.columns
folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof_pca = np.zeros(len(train_x_2))
predictions_all = np.zeros(len(test_x_2))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x_2.values, train_y)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train_x_2.iloc[trn_idx][features],
                           label=train_y[trn_idx],
                           #categorical_feature=cat_feats
                           )
    val_data = lgb.Dataset(train_x_2.iloc[val_idx][features],
                           label=train_y[val_idx],
                           #categorical_feature=cat_feats
                           )

    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds=200)

    oof_pca[val_idx] = clf.predict(train_x_2.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions_all += clf.predict(test_x_2[features], num_iteration=clf.best_iteration) / folds.n_splits

print("RMSE test PCA: {:<8.5f}".format(mean_squared_error(predictions_all, test_y) ** 0.5))

