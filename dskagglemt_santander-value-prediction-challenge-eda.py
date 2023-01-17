import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
train = pd.read_csv('../input/santander-value-prediction-challenge/train.csv')
train.shape
train.head()
train.tail()
train.info()
dtype_df = train.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
list(train.select_dtypes(['object']).columns)
int_f = list(train.select_dtypes(['int']).columns)

float_f = list(train.select_dtypes(['float']).columns)



len(int_f), len(float_f)
for f in float_f:

    train[f] = train[f].astype('float32')
dtype_df = train.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
train.isnull().sum()
missing_df = train.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')

missing_df
# randomly took one columns

train['d5308d8bc'].nunique()
unique_df = train.nunique().reset_index()

unique_df.columns = ["col_name", "unique_count"]
unique_df
constant_df = unique_df[unique_df["unique_count"]==1]

constant_df
# train['9fc776466'].nunique()

train['9fc776466'].unique()
# constant_df.col_name.tolist()

print('Original Shape of Train Dataset {}'.format(train.shape))

train.drop(constant_df.col_name.tolist(), axis = 1, inplace = True)

print('Shape after dropping Constant Columns from Train Dataset {}'.format(train.shape))
train.head()
train_ids_df = train['ID']

train_ids_df.head()
train.drop('ID', axis = 1, inplace = True)
train.head()
# k = 15 # Number of variables for heatmap.

# target = 'target'



# cols = train[int_f].corr().nlargest(k, target)[target].index



# cm = train[cols].corr()



# plt.figure(figsize = (10, 6))



# sns.heatmap(cm, annot = True, cmap = 'viridis')
from sklearn.feature_selection import SelectKBest, chi2
select_feature = SelectKBest(score_func=chi2, k = 5)
X = train.drop('target', axis = 1)

y = train['target']



X.shape, y.shape
# select_feature.fit_transform(X, y)
train['target'].describe()
plt.figure(figsize=(8,6))

plt.scatter(range(train.shape[0]), np.sort(train['target'].values))

plt.xlabel('Index --> ', fontsize=12)

plt.ylabel('Target --> ', fontsize=12)

plt.title("Target Distribution", fontsize=14)

plt.show()
plt.figure(figsize=(12,8))

sns.distplot(train["target"].values, bins=50, kde=True)

plt.xlabel('Target --> ', fontsize=12)

plt.title("Target Histogram", fontsize=14)

plt.show()
# Taking log of target variable and re-check the same...

plt.figure(figsize=(12,8))

sns.distplot( np.log1p(train["target"].values), bins=50, kde=True)

plt.xlabel('Target --> ', fontsize=12)

plt.title("Log of Target Histogram", fontsize=14)

plt.show()
# from scipy.stats import spearmanr

# import warnings

# warnings.filterwarnings("ignore")



# labels = []

# values = []

# for col in train.columns:

#     if col not in ["ID", "target"]:

#         labels.append(col)

#         values.append(spearmanr(train[col].values, train["target"].values)[0])

# corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})

# corr_df = corr_df.sort_values(by='corr_values')

 

# corr_df = corr_df[(corr_df['corr_values']>0.1) | (corr_df['corr_values']<-0.1)]

# ind = np.arange(corr_df.shape[0])

# width = 0.9

# fig, ax = plt.subplots(figsize=(12,30))

# rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='b')

# ax.set_yticks(ind)

# ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')

# ax.set_xlabel("Correlation coefficient")

# ax.set_title("Correlation coefficient of the variables")

# plt.show()
# cols_to_use = corr_df[(corr_df['corr_values']>0.11) | (corr_df['corr_values']<-0.11)].col_labels.tolist()



# temp_df = train[cols_to_use]

# corrmat = temp_df.corr(method='spearman')

# f, ax = plt.subplots(figsize=(20, 20))



# # Draw the heatmap using seaborn

# sns.heatmap(corrmat, vmax=1., square=True, cmap="YlGnBu", annot=True)

# plt.title("Important variables correlation map", fontsize=15)

# plt.show()
# ### Get the X and y variables for building model ###

# train_X = train.drop(["ID", "target"], axis=1)

# train_y = np.log1p(train["target"].values)
# from sklearn import ensemble

# model = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=20, max_features=0.5, n_jobs=-1, random_state=0)

# model.fit(train_X, train_y)



# ## plot the importances ##

# feat_names = train_X.columns.values

# importances = model.feature_importances_

# std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

# indices = np.argsort(importances)[::-1][:20]



# plt.figure(figsize=(12,12))

# plt.title("Feature importances")

# plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")

# plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')

# plt.xlim([-1, len(indices)])

# plt.show()
from sklearn.model_selection import train_test_split 

  

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 0) 
X.shape
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
from sklearn.preprocessing import StandardScaler 

sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 

X_valid = sc.transform(X_valid)
from sklearn.decomposition import PCA 

  

pca = PCA(n_components = 10) 
X_train_pca = pca.fit_transform(X_train) 

X_valid_pca = pca.transform(X_valid) 
explained_variance = pca.explained_variance_ratio_ 
explained_variance
X_train_pca.shape, X_valid_pca.shape
len(pca.components_)
# df_comp = pd.DataFrame(pca.components_, X.columns) 

  

# plt.figure(figsize =(14, 6)) 

  

# # plotting heatmap 

# sns.heatmap(df_comp) 
from xgboost import XGBRegressor
clf_xgb = XGBRegressor()

clf_xgb.fit(X_train_pca, y_train)
y_pred = abs(clf_xgb.predict(X_valid_pca))
# import math



# #A function to calculate Root Mean Squared Logarithmic Error (RMSLE)

# def rmsle(y, y_pred):

#     assert len(y) == len(y_pred)

#     terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

#     return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
# rmsle_value = rmsle(y_valid, y_pred)

# rmsle_value
from sklearn.metrics import mean_squared_log_error

np.sqrt(mean_squared_log_error( y_valid, y_pred ))
y_valid.describe()
min(y_pred), max(y_pred)
test_df = pd.read_csv("../input/santander-value-prediction-challenge/test.csv")
test_df.shape
test_df.head()
float_f = list(test_df.select_dtypes(['float']).columns)
for f in float_f:

    test_df[f] = test_df[f].astype('float32')
dtype_test_df = train.dtypes.reset_index()

dtype_test_df.columns = ["Count", "Column Type"]

dtype_test_df.groupby("Column Type").aggregate('count').reset_index()
test_df.drop(constant_df.col_name.tolist(), axis = 1, inplace = True)

test_df.shape
test_ids_df = test_df['ID']

test_ids_df.head()
test_df.drop('ID', axis = 1, inplace = True)

test_df.shape
test_df = sc.transform(test_df)
test_df_pca = pca.transform(test_df) 
test_df_pca.shape
test_df_pca[:5]
test_df_pca_5 = test_df_pca[:5]
pred_test_full = abs(clf_xgb.predict(test_df_pca_5))

pred_test_full
pred_test_full = abs(clf_xgb.predict(test_df_pca))

# pred_test_full
len(pred_test_full)
# pred_test_full = []



# pr = clf_xgb.predict(test_df_pca_5)

# # print(len(pr))



# pred_test_full.append(pr.tolist())



# # pr

# # print(len(pred_test_full))

# # pred_test_full



# pr = clf_xgb.predict(test_df_pca[5:10])

# pred_test_full.append(pr.tolist())



# # print(len(pred_test_full))



# pred_test_full
# flat_list = [item for sublist in pred_test_full for item in sublist]

# flat_list
# pr.tolist()
# test_df_pca[0]

# np.reshape(test_df_pca[0],(1, test_df_pca[0].size))
# pred_test_full = []



# # for i in range(0,50) :

# for i in range(0,test_df_pca.shape[0]) :

#     pr = clf_xgb.predict(np.reshape(test_df_pca[i],(1, test_df_pca[i].size)))

#     pred_test_full.append(pr.tolist())

    

# flat_list = [item for sublist in pred_test_full for item in sublist]

# len(flat_list)
# l = train.shape[0]

# c = l



# pred_test_full = []



# for i in range(0, len(l), 20):

#     pred_test_full.append(test_df_pca[i:c])

#     c += l
# flat_list = [item for sublist in pred_test_full for item in sublist]

# len(flat_list)
# import lightgbm as lgb

# import xgboost as xgb

# from catboost import CatBoostRegressor
def run_lgb(train_X, train_y, val_X, val_y, test_X):

    params = {

        "objective" : "regression",

        "metric" : "rmse",

        "num_leaves" : 40,

        "learning_rate" : 0.004,

        "bagging_fraction" : 0.6,

        "feature_fraction" : 0.6,

        "bagging_frequency" : 6,

        "bagging_seed" : 42,

        "verbosity" : -1,

        "seed": 42

    }

    

    lgtrain = lgb.Dataset(train_X, label=train_y)

    lgval = lgb.Dataset(val_X, label=val_y)

    evals_result = {}

    model = lgb.train(params, lgtrain, 5000, 

                      valid_sets=[lgtrain, lgval], 

                      early_stopping_rounds=100, 

                      verbose_eval=150, 

                      evals_result=evals_result)

    

    pred_test_y = np.expm1(model.predict(test_X, num_iteration=model.best_iteration))

    return pred_test_y, model, evals_result
# # Training LGB

# pred_lgb, model, evals_result = run_lgb(X_train_pca, y_train, X_valid_pca, y_valid, test_df_pca)

# print("LightGBM Training Completed...")
# # feature importance

# print("Features Importance...")

# gain = model.feature_importance('gain')

# featureimp = pd.DataFrame({'feature':model.feature_name(), 

#                    'split':model.feature_importance('split'), 

#                    'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)

# print(featureimp[:50])
def run_xgb(train_X, train_y, val_X, val_y, test_X):

    params = {'objective': 'reg:linear', 

          'eval_metric': 'rmse',

          'eta': 0.001,

          'max_depth': 10, 

          'subsample': 0.6, 

          'colsample_bytree': 0.6,

          'alpha':0.001,

          'random_state': 42, 

          'silent': True}

    

    tr_data = xgb.DMatrix(train_X, train_y)

    va_data = xgb.DMatrix(val_X, val_y)

    

    watchlist = [(tr_data, 'train'), (va_data, 'valid')]

    

    model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 100, verbose_eval=100)

    

    dtest = xgb.DMatrix(test_X)

    xgb_pred_y = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))

    

    return xgb_pred_y, model_xgb
# # Training XGB

# pred_test_xgb, model_xgb = run_xgb(X_train_pca, y_train, X_valid_pca, y_valid, test_df_pca)

# print("XGB Training Completed...")
# cb_model = CatBoostRegressor(iterations=500,

#                              learning_rate=0.05,

#                              depth=10,

#                              eval_metric='RMSE',

#                              random_seed = 42,

#                              bagging_temperature = 0.2,

#                              od_type='Iter',

#                              metric_period = 50,

#                              od_wait=20)
# cb_model.fit(X_train_pca, y_train,

#              eval_set=(X_valid_pca, y_valid),

#              use_best_model=True,

#              verbose=50)

 
# pred_test_cat = np.expm1(cb_model.predict(test_df_pca))
# from sklearn.metrics import mean_squared_log_error

# np.sqrt(mean_squared_log_error( y_valid, y_pred ))
subm_sample = pd.read_csv('../input/santander-value-prediction-challenge/sample_submission.csv')

subm_sample.head()
# test_ids_df.head()

test_ids_df[:]
subm_df = pd.DataFrame({"ID":test_ids_df[:]})

subm_df["target"] = pred_test_full

subm_df.to_csv("XGBReg_v1.csv", index=False)
subm_df.head()