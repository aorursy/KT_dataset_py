import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

%matplotlib inline

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb
import xgboost as xgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
IS_LOCAL=False
if(IS_LOCAL):
    PATH="../input/google-analytics-customer-revenue-prediction/"    
else:
    PATH="../input/"

#the columns that will be parsed to extract the fields from the jsons
cols_to_parse = ['device', 'geoNetwork', 'totals', 'trafficSource']
def read_parse_dataframe(file_name):
    #full path for the data file
    path = PATH + file_name
    #read the data file, convert the columns in the list of columns to parse using json loader,
    #convert the `fullVisitorId` field as a string
    data_df = pd.read_csv(path, 
        converters={column: json.loads for column in cols_to_parse}, 
        dtype={'fullVisitorId': 'str'})
    #parse the json-type columns
    for col in cols_to_parse:
        #each column became a dataset, with the columns the fields of the Json type object
        json_col_df = json_normalize(data_df[col])
        json_col_df.columns = [f"{col}_{sub_col}" for sub_col in json_col_df.columns]
        #we drop the object column processed and we add the columns created from the json fields
        data_df = data_df.drop(col, axis=1).merge(json_col_df, right_index=True, left_index=True)
    return data_df

def process_date_time(data_df):
    print("process date time ...")
    data_df['date'] = data_df['date'].astype(str)
    data_df["date"] = data_df["date"].apply(lambda x : x[:4] + "-" + x[4:6] + "-" + x[6:])
    data_df["date"] = pd.to_datetime(data_df["date"])   
    data_df["year"] = data_df['date'].dt.year
    data_df["month"] = data_df['date'].dt.month
    data_df["day"] = data_df['date'].dt.day
    data_df["weekday"] = data_df['date'].dt.weekday
    data_df['weekofyear'] = data_df['date'].dt.weekofyear
    data_df['month_unique_user_count'] = data_df.groupby('month')['fullVisitorId'].transform('nunique')
    data_df['day_unique_user_count'] = data_df.groupby('day')['fullVisitorId'].transform('nunique')
    data_df['weekday_unique_user_count'] = data_df.groupby('weekday')['fullVisitorId'].transform('nunique')
    return data_df

def process_format(data_df):
    print("process format ...")
    for col in ['visitNumber', 'totals_hits', 'totals_pageviews']:
        data_df[col] = data_df[col].astype(float)
    data_df['trafficSource_adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    data_df['trafficSource_isTrueDirect'].fillna(False, inplace=True)
    return data_df
    
def process_device(data_df):
    print("process device ...")
    data_df['browser_category'] = data_df['device_browser'] + '_' + data_df['device_deviceCategory']
    data_df['browser_operatingSystem'] = data_df['device_browser'] + '_' + data_df['device_operatingSystem']
    data_df['source_country'] = data_df['trafficSource_source'] + '_' + data_df['geoNetwork_country']
    return data_df

def process_totals(data_df):
    print("process totals ...")
    data_df['visitNumber'] = np.log1p(data_df['visitNumber'])
    data_df['totals_hits'] = np.log1p(data_df['totals_hits'])
    data_df['totals_pageviews'] = np.log1p(data_df['totals_pageviews'].fillna(0))
    data_df['mean_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('mean')
    data_df['sum_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('sum')
    data_df['max_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('max')
    data_df['min_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('min')
    data_df['var_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('var')
    return data_df

def process_geo_network(data_df):
    print("process geo network ...")
    data_df['sum_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('sum')
    data_df['count_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('count')
    data_df['mean_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('mean')
    data_df['sum_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('sum')
    data_df['count_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('count')
    data_df['mean_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('mean')
    return data_df
train_df = read_parse_dataframe('../input/train.csv')
train_df = process_date_time(train_df)
test_df = read_parse_dataframe('../input/test.csv')
test_df = process_date_time(test_df)
cols_to_drop = [col for col in train_df.columns if train_df[col].nunique(dropna=False) == 1]
train_df.drop(cols_to_drop, axis=1, inplace=True)
test_df.drop([col for col in cols_to_drop if col in test_df.columns], axis=1, inplace=True)
train_df.drop(['trafficSource_campaignCode'], axis=1, inplace=True)
train_df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].astype(float)
train_df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].fillna(0)
train_df['totals_transactionRevenue'] = np.log1p(train_df['totals_transactionRevenue'])
train_df = process_format(train_df)
train_df = process_device(train_df)
train_df = process_totals(train_df)
train_df = process_geo_network(train_df)

test_df = process_format(test_df)
test_df = process_device(test_df)
test_df = process_totals(test_df)
test_df = process_geo_network(test_df)
from sklearn.preprocessing import LabelEncoder
print("process categorical columns ...")
num_cols = ['month_unique_user_count', 'day_unique_user_count', 'weekday_unique_user_count',
            'visitNumber', 'totals_hits', 'totals_pageviews', 
            'mean_hits_per_day', 'sum_hits_per_day', 'min_hits_per_day', 'max_hits_per_day', 'var_hits_per_day',
            'sum_pageviews_per_network_domain', 'count_pageviews_per_network_domain', 'mean_pageviews_per_network_domain',
            'sum_hits_per_network_domain', 'count_hits_per_network_domain', 'mean_hits_per_network_domain']
            
not_used_cols = ["visitNumber", "date", "fullVisitorId", "sessionId", 
        "visitId", "visitStartTime", 'totals_transactionRevenue', 'trafficSource_referralPath']
cat_cols = [col for col in train_df.columns if col not in num_cols and col not in not_used_cols]
for col in cat_cols:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
# Model
print("prepare model ...")
train_df = train_df.sort_values('date')
X = train_df.drop(not_used_cols, axis=1)
y = train_df['totals_transactionRevenue']
X_test = test_df.drop([col for col in not_used_cols if col in test_df.columns], axis=1)
print("Number of unique visitors in train set : ",train_df.fullVisitorId.nunique(), " out of rows : ",train_df.shape[0])
print("Number of unique visitors in train set : ",test_df.fullVisitorId.nunique(), " out of rows : ",test_df.shape[0])
print("Number of common visitors in train and test set : ",len(set(train_df.fullVisitorId.unique()).intersection(set(test_df.fullVisitorId.unique())) ))
plt.figure(figsize=(20,20))
sns.heatmap(train_df.corr(),annot=True)
train_df_new = train_df

#print(train_df_new)

#Now let us look at the correlation coefficient of each of these variables #
x_cols = [col for col in train_df_new.columns if col not in ['totals_transactionRevenue'] if train_df_new[col].dtype=='float64']

labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(train_df_new[col].values, train_df_new.totals_transactionRevenue.values)[0,1])
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
# corr_df_sel = corr_df.ix[(corr_df['corr_values']>0.05)]
# corr_df_sel['col_labels'].values
col_x = ['sum_pageviews_per_network_domain', 'sum_hits_per_network_domain',
       'mean_hits_per_network_domain',
       'mean_pageviews_per_network_domain', 'totals_hits',
       'totals_pageviews']

for i in col_x:
    X[i+'_square'] =  X[i] ** 2
    X_test[i+'_square'] = X_test[i] ** 2
X.shape,X_test.shape
[c for c in train_df.columns if train_df[c].nunique()==1]
print("Variables not in test but in train : ", set(train_df.columns).difference(set(test_df.columns)))
from sklearn.model_selection import KFold
# import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# params = {"objective" : "regression", "metric" : "rmse", "max_depth": 8, "min_child_samples": 20, "reg_alpha": 1, "reg_lambda": 1,'min_data_in_leaf': 20,
#         "num_leaves" : 350, "learning_rate" : 0.01, "subsample" : 0.8, "colsample_bytree" : 0.85, "subsample_freq ": 5, "feature_fraction":0.95,
#          "bagging_freq":1,"bagging_fraction":0.85 ,"bagging_seed": 32,"lambda_l1": 0.89}
params = {"objective" : "regression", "metric" : "rmse","num_leaves" : 30, "learning_rate" : 0.01, "bagging_fraction" : 0.9,"feature_fraction" : 0.3, "bagging_seed" : 0}
folds = KFold(n_splits=10, shuffle=True, random_state=42)
NUM_ROUNDS = 20000
VERBOSE_EVAL = 1000
STOP_ROUNDS = 100
print("lgb_model ...")
lgb_model = lgb.LGBMRegressor(**params, n_estimators = NUM_ROUNDS, nthread = 4, n_jobs = -1)

prediction = np.zeros(test_df.shape[0])

for fold_n, (train_index, test_index) in enumerate(folds.split(X)):
    print('Fold:', fold_n)
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    lgb_model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
            verbose=VERBOSE_EVAL, early_stopping_rounds=STOP_ROUNDS)
    
    y_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration_)
    prediction += y_pred
prediction /= 10
fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(lgb_model, max_num_features=50, height=0.8, ax=ax)
lgb_features = lgb_model.feature_importances_
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=20)
plt.show()
# Submission
print("prepare submission ...")
submission = test_df[['fullVisitorId']].copy()
submission.loc[:, 'PredictedLogRevenue'] = prediction
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test.to_csv('lgb.csv',index=False)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
folds1 = KFold(n_splits=10, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=1)

print(f"Train shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test (submit) shape: {X_test.shape}")
print("model ...")
params1 = {
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'eta': 0.001,
            'max_depth': 7,
            'gamma': 1.3250360141843498, 
            'min_child_weight': 13.0958516960316, 
            'max_delta_step': 8.88492863796954, 
            'subsample': 0.9864199446951019, 
            'colsample_bytree': 0.8376539278239742,
            'subsample': 0.6,
            'colsample_bytree': 0.8,
            'alpha':0.001,
            "num_leaves" : 40,
            'random_state': 42,
            'silent': True,
            }


#  params = {'max_depth': 7, 
#            'gamma': 1.3250360141843498, 
#            'min_child_weight': 13.0958516960316, 
#            'max_delta_step': 8.88492863796954, 
#            'subsample': 0.9864199446951019, 
#            'colsample_bytree': 0.8376539278239742}

prediction = np.zeros(test_df.shape[0])

for fold_n1, (train_index, test_index) in enumerate(folds1.split(X)):
    print('Fold:', fold_n1)
    xgb_train_data = xgb.DMatrix(X_train, y_train)
    xgb_val_data = xgb.DMatrix(X_val, y_val)
    xgb_submit_data = xgb.DMatrix(X_test)

    xgb_model = xgb.train(params1, xgb_train_data, 
                      # Note: I disabled XGB to make the notebook run faster
                      # Set to 2000 to obtain the results reported in Conclusion
                      num_boost_round=1000, 
                      evals= [(xgb_train_data, 'train'), (xgb_val_data, 'valid')],
                      early_stopping_rounds=100, 
                      verbose_eval=500
                     )
y_pred = xgb_model.predict(xgb_submit_data, ntree_limit=model.best_ntree_limit)
prediction += y_pred
prediction /= 10
fig, ax = plt.subplots(figsize=(12,18))
xgb_model.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
xgb_model.feature_importance_
ax.grid(False)
plt.title("XGB - Feature Importance", fontsize=20)
plt.show()
xgb.plot_tree(model,num_trees=0, rankdir='LR')
xgb.to_graphviz(model, num_trees=2)
# Submission
print("prepare submission ...")
submission = test_df[['fullVisitorId']].copy()
submission.loc[:, 'PredictedLogRevenue'] = prediction
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test.to_csv('xgb.csv',index=False)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
folds2 = KFold(n_splits=10, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=1)

print(f"Train shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test (submit) shape: {X_test.shape}")
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 5)
print("model ...")

prediction = np.zeros(test_df.shape[0])
model = CatBoostRegressor(iterations=1000,
                             learning_rate=0.05,
                             depth=10,
                             eval_metric='RMSE',
                             random_seed = 42,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)

for fold_n2, (train_index, test_index) in enumerate(folds2.split(X)):
    print('Fold:', fold_n2)
    model.fit(X_train, y_train,eval_set=(X_val, y_val),use_best_model=True,verbose=True)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_submit = model.predict(X_test)
    print(f"CatB: RMSE val: {rmse(y_val, y_pred_val)}  - RMSE train: {rmse(y_train, y_pred_train)}")
y_pred_submit = model.predict(X_test)
prediction += y_pred_submit
prediction /= 10
feat_imp = pd.DataFrame({'importance':model.feature_importances_})    
feat_imp['feature'] = X_train.columns
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
feat_imp.sort_values(by='importance', inplace=True)
feat_imp = feat_imp.set_index('feature', drop=True)
feat_imp.plot.barh(title='Cat Boost Feature Importance', figsize = (20,40), fontsize=20)
plt.xlabel('Feature Importance Score')
plt.show()
# Submission
print("prepare submission ...")
submission = test_df[['fullVisitorId']].copy()
submission.loc[:, 'PredictedLogRevenue'] = prediction
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test.to_csv('catb.csv',index=False)
# Note: I disabled XGB to make the notebook run faster, it was 70-25-5 before
lgb = pd.read_csv("lgb.csv")
xgb = pd.read_csv("xgb.csv")
catb = pd.read_csv("catb.csv")
submission = pd.read_csv("../input/sample_submission.csv")

submission["PredictedLogRevenue"] = lgb["PredictedLogRevenue"] * 0.6 + catb["PredictedLogRevenue"] * 0.3 + xgb["PredictedLogRevenue"] * 0.1

submission.to_csv("Mix_model.csv", index = False)
