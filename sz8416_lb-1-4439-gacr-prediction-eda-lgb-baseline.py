import os
import time
import gc
import warnings
warnings.filterwarnings("ignore")
# data manipulation
import json
from pandas.io.json import json_normalize
import numpy as np
import pandas as pd
# plot
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
# model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
%%time
def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
train = load_df('../input/train.csv')
test = load_df('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')
gc.collect()
train.head()
set(train.columns).difference(set(test.columns))
cons_col = [i for i in train.columns if train[i].nunique(dropna=False)==1]
cons_col
train = train.drop(cons_col + ['trafficSource.campaignCode'], axis=1)
test = test.drop(cons_col, axis=1)
gc.collect()
print(train.shape)
print(test.shape)
def find_missing(data):
    # number of missing values
    count_missing = data.isnull().sum().values
    # total records
    total = data.shape[0]
    # percentage of missing
    ratio_missing = count_missing/total
    # return a dataframe to show: feature name, # of missing and % of missing
    return pd.DataFrame(data={'missing_count':count_missing, 'missing_ratio':ratio_missing}, index=data.columns.values)
train_missing = find_missing(train)
test_missing = find_missing(test)
train_missing.reset_index()[['index', 'missing_ratio']]\
    .merge(test_missing.reset_index()[['index', 'missing_ratio']], on='index', how='left')\
    .rename(columns={'index':'columns', 'missing_ratio_x':'train_missing_ratio', 'missing_ratio_y':'test_missing_ratio'})\
    .sort_values(['train_missing_ratio', 'test_missing_ratio'], ascending=False)\
    .query('train_missing_ratio>0')
if test.fullVisitorId.nunique() == len(sub):
    print('Till now, the number of fullVisitorId is equal to the rows in submission. Everything goes well!')
else:
    print('Check it again')
y = np.nan_to_num(np.array([float(i) for i in train['totals.transactionRevenue']]))
print('The ratio of customers with transaction revenue is', str((y != 0).mean()))   
plt.figure(figsize=[12, 6])
sns.distplot(y[y!=0])
plt.xlabel('transactionRevenue')
plt.show()
train["totals.transactionRevenue"] = train["totals.transactionRevenue"].astype('float')
target = np.log1p(train.groupby("fullVisitorId")["totals.transactionRevenue"].sum())
print('The ratio of customers with transaction revenue is', str((target != 0).mean()))
plt.figure(figsize=[12, 6])
sns.distplot(target[target!=0])
plt.xlabel('Target')
plt.show()
def plot_categorical(data, col, size=[8 ,4], xlabel_angle=0, title='', max_cat = None):
    '''use this for ploting the count of categorical features'''
    plotdata = data[col].value_counts() / len(data)
    if max_cat != None:
        plotdata = plotdata[max_cat[0]:max_cat[1]]
    plt.figure(figsize = size)
    sns.barplot(x = plotdata.index, y=plotdata.values)
    plt.title(title)
    if xlabel_angle!=0: 
        plt.xticks(rotation=xlabel_angle)
    plt.show()
plot_categorical(data=train, col='device.browser', size=[8 ,4], xlabel_angle=20, title='Device - Browser', max_cat=[0, 6])
plot_categorical(data=train, col='device.deviceCategory', size=[8 ,4], xlabel_angle=0, title='Device - Category')
plot_categorical(data=train, col='device.operatingSystem', size=[8 ,4], xlabel_angle=30, 
                 title='Device - Operating System', max_cat = [0, 7])
plot_categorical(data=train, col='geoNetwork.city', size=[12 ,4], xlabel_angle=30, 
                 title='GeoNetwork - City', max_cat = [1, 20])
plot_categorical(data=train, col='geoNetwork.country', size=[12 ,4], xlabel_angle=30, 
                 title='GeoNetwork - Country', max_cat = [0, 20])
plot_categorical(data=train, col='geoNetwork.region', size=[12 ,4], xlabel_angle=30, 
                 title='GeoNetwork - Region', max_cat = [1, 20])
plot_categorical(data=train, col='geoNetwork.metro', size=[12 ,4], xlabel_angle=90, 
                 title='GeoNetwork - metro', max_cat = [2, 20])
plot_categorical(data=train, col='geoNetwork.subContinent', size=[8 ,4], xlabel_angle=30, 
                 title='GeoNetwork - SubContinent', max_cat = [0, 10])
plot_categorical(data=train, col='geoNetwork.continent', size=[8 ,4], xlabel_angle=30, 
                 title='GeoNetwork - Continent')
train['totals.bounces'] = train['totals.bounces'].fillna('0')
plot_categorical(data=train, col='totals.bounces', size=[8 ,4], xlabel_angle=0, title='Totals - Bounces')
train['totals.newVisits'] = train['totals.newVisits'].fillna('0')
plot_categorical(data=train, col='totals.newVisits', size=[8 ,4], xlabel_angle=0, title='Totals - NewVisits')
plt.figure(figsize=[12, 6])
sns.distplot(train['totals.hits'].astype('float'), kde=True,bins=30)
plt.xlabel('totals.hits')
plt.title('Total - Hits')
plt.show()
plt.figure(figsize=[12, 6])
sns.distplot(train['totals.pageviews'].astype('float').fillna(0))
plt.xlabel('totals.pageviews')
plt.title('Total - Pageviews')
plt.show()
plot_categorical(data=train, col='trafficSource.adContent', size=[10 ,4], xlabel_angle=30, 
                 title='TrafficSource - AdContent', max_cat = [0, 10])
plot_categorical(data=train, col='trafficSource.medium', size=[10 ,4], xlabel_angle=30, 
                 title='TrafficSource - medium')
plot_categorical(data=train, col='channelGrouping', size=[10 ,4], xlabel_angle=30, 
                 title='Channel Grouping')
a = train.groupby("fullVisitorId")["visitNumber"].max()
plt.figure(figsize=[12, 6])
sns.distplot(a)
plt.xlabel('VisitNumber')
plt.title('Visit Number')
plt.show()
plt.figure(figsize=[12, 6])
sns.distplot(train['date'])
plt.xlabel('Date')
plt.title('Date')
plt.show()
train_idx = train.fullVisitorId
test_idx = test.fullVisitorId
train["totals.transactionRevenue"] = train["totals.transactionRevenue"].astype('float').fillna(0)
train_y = train["totals.transactionRevenue"]
train_target = np.log1p(train.groupby("fullVisitorId")["totals.transactionRevenue"].sum())
train.drop(['fullVisitorId', 'sessionId', 'visitId'], axis = 1, inplace = True)
test.drop(['fullVisitorId', 'sessionId', 'visitId'], axis = 1, inplace = True)
num_col = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits']
for i in num_col:
    train[i] = train[i].astype('float').fillna(0)
    test[i] = test[i].astype('float').fillna(0)
cat_col = [e for e in train.columns.tolist() if e not in num_col]
cat_col.remove('date')
cat_col.remove('totals.transactionRevenue')
for i in cat_col:
    lab_en = LabelEncoder()
    train[i] = train[i].fillna('not known')
    test[i] = test[i].fillna('not known')
    lab_en.fit(list(train[i].astype('str')) + list(test[i].astype('str')))
    train[i] = lab_en.transform(list(train[i].astype('str')))
    test[i] = lab_en.transform(test[i].astype('str'))
    print('finish', i)
y_train = np.log1p(train["totals.transactionRevenue"])
x_train = train.drop(["totals.transactionRevenue"], axis=1)
x_test = test.copy()
print(x_train.shape)
print(x_test.shape)
folds = KFold(n_splits=5,random_state=6)
oof_preds = np.zeros(x_train.shape[0])
sub_preds = np.zeros(x_test.shape[0])

start = time.time()
valid_score = 0
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
    trn_x, trn_y = x_train.iloc[trn_idx], y_train[trn_idx]
    val_x, val_y = x_train.iloc[val_idx], y_train[val_idx]    
    
    train_data = lgb.Dataset(data=trn_x, label=trn_y)
    valid_data = lgb.Dataset(data=val_x, label=val_y)
    
    params = {"objective" : "regression", "metric" : "rmse", 'n_estimators':10000, 'early_stopping_rounds':100,
              "num_leaves" : 30, "learning_rate" : 0.01, "bagging_fraction" : 0.9,
              "feature_fraction" : 0.3, "bagging_seed" : 0}
    
    lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000) 
    
    oof_preds[val_idx] = lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration)
    oof_preds[oof_preds<0] = 0
    sub_pred = lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration) / folds.n_splits
    sub_pred[sub_pred<0] = 0 # should be greater or equal to 0
    sub_preds += sub_pred
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, np.sqrt(mean_squared_error(val_y, oof_preds[val_idx]))))
    valid_score += np.sqrt(mean_squared_error(val_y, oof_preds[val_idx]))
print('Session-level CV-score:', str(round(valid_score/folds.n_splits,4)))
print(' ')
train_pred = pd.DataFrame({"fullVisitorId":train_idx})
train_pred["PredictedLogRevenue"] = np.expm1(oof_preds)
train_pred = train_pred.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
train_pred.columns = ["fullVisitorId", "PredictedLogRevenue"]
train_pred["PredictedLogRevenue"] = np.log1p(train_pred["PredictedLogRevenue"])
train_rmse = np.sqrt(mean_squared_error(train_target, train_pred['PredictedLogRevenue']))
print('User-level score:', str(round(train_rmse, 4)))
print(' ')
end = time.time()
print('training time:', str(round((end - start)/60)), 'mins')
test_pred = pd.DataFrame({"fullVisitorId":test_idx})
test_pred["PredictedLogRevenue"] = np.expm1(sub_preds)
test_pred = test_pred.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
test_pred.columns = ["fullVisitorId", "PredictedLogRevenue"]
test_pred["PredictedLogRevenue"] = np.log1p(test_pred["PredictedLogRevenue"])
test_pred.to_csv("lgb_base_model.csv", index=False) # submission
lgb.plot_importance(lgb_model, height=0.5, max_num_features=20, ignore_zero = False, figsize = (12,6), importance_type ='gain')
plt.show()