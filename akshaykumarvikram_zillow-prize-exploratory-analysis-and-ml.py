import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import ggplot

from ggplot import aes

color = sns.color_palette()



import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

import gc

from sklearn.linear_model import LinearRegression

import random

import datetime as dt



%matplotlib inline



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_df = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])

train_df.shape
train_df.head()
plt.figure(figsize=(8,6))

plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('logerror', fontsize=12)

plt.show()
ulimit = np.percentile(train_df.logerror.values, 99)

llimit = np.percentile(train_df.logerror.values, 1)

train_df['logerror'].ix[train_df['logerror']>ulimit] = ulimit

train_df['logerror'].ix[train_df['logerror']<llimit] = llimit



plt.figure(figsize=(12,8))

sns.distplot(train_df.logerror.values, bins=50, kde=False)

plt.xlabel('logerror', fontsize=12)

plt.show()
train_df['transaction_month'] = train_df['transactiondate'].dt.month



cnt_srs = train_df['transaction_month'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Month of transaction', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()
(train_df['parcelid'].value_counts().reset_index())['parcelid'].value_counts()
prop_df = pd.read_csv("../input/properties_2016.csv")

prop_df.shape
prop_df.head()
missing_df = prop_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.ix[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')



ind = np.arange(missing_df.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missing_df.missing_count.values, color='blue')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
plt.figure(figsize=(12,12))

sns.jointplot(x=prop_df.latitude.values, y=prop_df.longitude.values, size=10)

plt.ylabel('Longitude', fontsize=12)

plt.xlabel('Latitude', fontsize=12)

plt.show()
train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')

train_df.head()
pd.options.display.max_rows = 65



dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df
dtype_df.groupby("Column Type").aggregate('count').reset_index()
missing_df = train_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df['missing_ratio'] = missing_df['missing_count'] / train_df.shape[0]

missing_df.ix[missing_df['missing_ratio']>0.999]
# Let us just impute the missing values with mean values to compute correlation coefficients #

mean_values = train_df.mean(axis=0)

train_df_new = train_df.fillna(mean_values, inplace=True)



# Now let us look at the correlation coefficient of each of these variables #

x_cols = [col for col in train_df_new.columns if col not in ['logerror'] if train_df_new[col].dtype=='float64']



labels = []

values = []

for col in x_cols:

    labels.append(col)

    values.append(np.corrcoef(train_df_new[col].values, train_df_new.logerror.values)[0,1])

corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})

corr_df = corr_df.sort_values(by='corr_values')

    

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots(figsize=(12,40))

rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')

ax.set_yticks(ind)

ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')

ax.set_xlabel("Correlation coefficient")

ax.set_title("Correlation coefficient of the variables")

#autolabel(rects)

plt.show()
corr_zero_cols = ['assessmentyear', 'storytypeid', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10', 'poolcnt', 'decktypeid', 'buildingclasstypeid']

for col in corr_zero_cols:

    print(col, len(train_df_new[col].unique()))
corr_df_sel = corr_df.ix[(corr_df['corr_values']>0.02) | (corr_df['corr_values'] < -0.01)]

corr_df_sel
cols_to_use = corr_df_sel.col_labels.tolist()



temp_df = train_df[cols_to_use]

corrmat = temp_df.corr(method='spearman')

f, ax = plt.subplots(figsize=(8, 8))



# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=1., square=True)

plt.title("Important variables correlation map", fontsize=15)

plt.show()
col = "finishedsquarefeet12"

ulimit = np.percentile(train_df[col].values, 99.5)

llimit = np.percentile(train_df[col].values, 0.5)

train_df[col].ix[train_df[col]>ulimit] = ulimit

train_df[col].ix[train_df[col]<llimit] = llimit



plt.figure(figsize=(12,12))

sns.jointplot(x=train_df.finishedsquarefeet12.values, y=train_df.logerror.values, size=10, color=color[4])

plt.ylabel('Log Error', fontsize=12)

plt.xlabel('Finished Square Feet 12', fontsize=12)

plt.title("Finished square feet 12 Vs Log error", fontsize=15)

plt.show()
col = "calculatedfinishedsquarefeet"

ulimit = np.percentile(train_df[col].values, 99.5)

llimit = np.percentile(train_df[col].values, 0.5)

train_df[col].ix[train_df[col]>ulimit] = ulimit

train_df[col].ix[train_df[col]<llimit] = llimit



plt.figure(figsize=(12,12))

sns.jointplot(x=train_df.calculatedfinishedsquarefeet.values, y=train_df.logerror.values, size=10, color=color[5])

plt.ylabel('Log Error', fontsize=12)

plt.xlabel('Calculated finished square feet', fontsize=12)

plt.title("Calculated finished square feet Vs Log error", fontsize=15)

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(x="bathroomcnt", data=train_df)

plt.ylabel('Count', fontsize=12)

plt.xlabel('Bathroom', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of Bathroom count", fontsize=15)

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(x="bathroomcnt", y="logerror", data=train_df)

plt.ylabel('Log error', fontsize=12)

plt.xlabel('Bathroom Count', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("How log error changes with bathroom count?", fontsize=15)

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(x="bedroomcnt", data=train_df)

plt.ylabel('Frequency', fontsize=12)

plt.xlabel('Bedroom Count', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of Bedroom count", fontsize=15)

plt.show()
train_df['bedroomcnt'].ix[train_df['bedroomcnt']>7] = 7

plt.figure(figsize=(12,8))

sns.violinplot(x='bedroomcnt', y='logerror', data=train_df)

plt.xlabel('Bedroom count', fontsize=12)

plt.ylabel('Log Error', fontsize=12)

plt.show()
col = "taxamount"

ulimit = np.percentile(train_df[col].values, 99.5)

llimit = np.percentile(train_df[col].values, 0.5)

train_df[col].ix[train_df[col]>ulimit] = ulimit

train_df[col].ix[train_df[col]<llimit] = llimit



plt.figure(figsize=(12,12))

sns.jointplot(x=train_df['taxamount'].values, y=train_df['logerror'].values, size=10, color='g')

plt.ylabel('Log Error', fontsize=12)

plt.xlabel('Tax Amount', fontsize=12)

plt.title("Tax Amount Vs Log error", fontsize=15)

plt.show()
train_y = train_df['logerror'].values

cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]

train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate', 'transaction_month']+cat_cols, axis=1)

feat_names = train_df.columns.values



from sklearn import ensemble

model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)

model.fit(train_df, train_y)



## plot the importances ##

importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

indices = np.argsort(importances)[::-1][:20]



plt.figure(figsize=(12,12))

plt.title("Feature importances")

plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')

plt.xlim([-1, len(indices)])

plt.show()
import xgboost as xgb

y_mean = np.mean(train_y)

xgb_params = {

    'eta': 0.037,

    'max_depth': 5,

    'subsample': 0.80,

    'objective': 'reg:linear',

    'eval_metric': 'mae',

    'lambda': 0.8,   

    'alpha': 0.4, 

    'base_score': y_mean,

    'silent': 1

}

dtrain = xgb.DMatrix(train_df, train_y, feature_names=train_df.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)



# plot the important features #

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
for c, dtype in zip(prop_df.columns, prop_df.dtypes):	

    if dtype == np.float64:		

        prop_df[c] = prop_df[c].astype(np.float32)
train_columns = train_df.columns

for c in train_df.dtypes[train_df.dtypes == object].index.values:

    train_df[c] = (train_df[c] == True)

train_df = train_df.values.astype(np.float32, copy=False)

params = {}

params['max_bin'] = 10

params['learning_rate'] = 0.0021 # shrinkage_rate

params['boosting_type'] = 'gbdt'

params['objective'] = 'regression'

params['metric'] = 'l1'          # or 'mae'

params['sub_feature'] = 0.345    # feature_fraction (small values => use very different submodels)

params['bagging_fraction'] = 0.85 # sub_row

params['bagging_freq'] = 40

params['num_leaves'] = 512        # num_leaf

params['min_data'] = 500         # min_data_in_leaf

params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf

params['verbose'] = 0

params['feature_fraction_seed'] = 2

params['bagging_seed'] = 3



np.random.seed(0)

random.seed(0)
d_train = lgb.Dataset(train_df, label=train_y)

clf = lgb.train(params, d_train, 430)
sample = pd.read_csv('../input/sample_submission.csv')

sample['parcelid'] = sample['ParcelId']

test_df = sample.merge(prop_df, on='parcelid', how='left')

x_test = test_df[train_columns]

for c in x_test.dtypes[x_test.dtypes == object].index.values:

    x_test[c] = (x_test[c] == True)

x_test = x_test.values.astype(np.float32, copy=False)

p_test = clf.predict(x_test)

pd.DataFrame(p_test).head()
xgb_params = {

    'eta': 0.037,

    'max_depth': 5,

    'subsample': 0.80,

    'objective': 'reg:linear',

    'eval_metric': 'mae',

    'lambda': 0.8,   

    'alpha': 0.4, 

    'base_score': y_mean,

    'silent': 1

}

dtrain = xgb.DMatrix(train_df, train_y)

dtest = xgb.DMatrix(x_test)
num_boost_rounds = 250

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

xgb_pred1 = model.predict(dtest)

pd.DataFrame(xgb_pred1).head()
xgb_params = {

    'eta': 0.033,

    'max_depth': 6,

    'subsample': 0.80,

    'objective': 'reg:linear',

    'eval_metric': 'mae',

    'base_score': y_mean,

    'silent': 1

}

num_boost_rounds = 150

model2 = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

xgb_pred2 = model.predict(dtest)

pd.DataFrame(xgb_pred2).head()

                   
# Parameters

XGB_WEIGHT = 0.6400

BASELINE_WEIGHT = 0.0056

OLS_WEIGHT = 0.0828

XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models

BASELINE_PRED = 0.0115
xgb_pred = XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*xgb_pred2

pd.DataFrame(xgb_pred).head()
lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)

xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)

baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)

pred0 = xgb_weight0*xgb_pred + baseline_weight0*BASELINE_PRED + lgb_weight*p_test
np.random.seed(17)

random.seed(17)



train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])

properties = pd.read_csv("../input/properties_2016.csv")

submission = pd.read_csv("../input/sample_submission.csv")
def get_features(df):

    df["transactiondate"] = pd.to_datetime(df["transactiondate"])

    df["transactiondate_year"] = df["transactiondate"].dt.year

    df["transactiondate_month"] = df["transactiondate"].dt.month

    df['transactiondate'] = df['transactiondate'].dt.quarter

    df = df.fillna(-1.0)

    return df



def MAE(y, ypred):

    #logerror=log(Zestimate)−log(SalePrice)

    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)
train = pd.merge(train, properties, how='left', on='parcelid')

y = train['logerror'].values

test = pd.merge(submission, properties, how='left', left_on='ParcelId', right_on='parcelid')

properties = [] #memory
exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','parcelid']

col = [c for c in train.columns if c not in exc]
train = get_features(train[col])

test['transactiondate'] = '2016-01-01' #should use the most common training date

test = get_features(test[col])
reg = LinearRegression(n_jobs=-1)

reg.fit(train, y); print('fit...')

print(MAE(y, reg.predict(train)))

train = [];  y = [] 
test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']

test_columns = ['201610','201611','201612','201710','201711','201712']

print( "\nCombining XGBoost, LightGBM, and baseline predicitons ..." )

lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)

xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)

baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)

pred0 = xgb_weight0*xgb_pred + baseline_weight0*BASELINE_PRED + lgb_weight*p_test

print( "\nCombined XGB/LGB/baseline predictions:" )

print( pd.DataFrame(pred0).head() )

print( "\nPredicting with OLS and combining with XGB/LGB/baseline predicitons: ..." )

for i in range(len(test_dates)):

    test['transactiondate'] = test_dates[i]

    pred = OLS_WEIGHT*reg.predict(get_features(test)) + (1-OLS_WEIGHT)*pred0

    submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]

    print('predict...', i)



print( "\nCombined XGB/LGB/baseline/OLS predictions:" )

print( submission.head() )
from datetime import datetime



print( "\nWriting results to disk ..." )

submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)



print( "\nFinished ...")
