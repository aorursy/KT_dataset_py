#import every library needed for completing task

import operator
import gc

import pandas as pd
import numpy as np
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import xgboost as xgb
import lightgbm as lgb
# make the functions which is needed

def regression_stats(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
    pearsonr = stats.pearsonr(x[mask], y[mask])
    result = {
        'slope': slope, 'intercept': intercept,
        'r_value': r_value, 'p_value': p_value,
        'std_err': std_err, 'r_squared': r_value ** 2,
        'pearsonr': pearsonr[0]
    }
    return result
# Acquire the dataset

sold_result = pd.read_csv('../input/train_2016_v2.csv')
prop = pd.read_csv('../input/properties_2016.csv', low_memory=False)

sold_result.shape, prop.shape
# change the dtypes of dataframes for using less memory

for c, dtype in zip(prop.columns, prop.dtypes):	
    if dtype == np.float64:	
        prop[c] = prop[c].astype(np.float32)
# train_result parcelid is not unique. I will make the meidan of it and will use the median as my training data

sold_result.parcelid.nunique(), prop.parcelid.nunique()
edited_sold_result = sold_result[['parcelid', 'logerror']].groupby('parcelid').agg(['median'])
edited_sold_result.columns = ['logerror']
edited_sold_result = edited_sold_result.reset_index()
edited_sold_result.head()
# make the final dataset to split into train dataset and test dataset

final_dataset = pd.merge(prop, edited_sold_result, how="left", on="parcelid")
final_dataset.shape
X_train  = final_dataset.loc[~final_dataset.logerror.isnull()]
y_train = final_dataset.loc[~final_dataset.logerror.isnull()].logerror.values
X_train.shape, y_train.shape
X_test = final_dataset.loc[final_dataset.logerror.isnull()]
X_test.shape
# make list of X_train, X_test to make it easier to do pre-processing the dataset laster

combine = [X_train, X_test]
pd.set_option('max_columns', 59)

X_train.tail()
# start to explore the dataset. For example, what kind of columns the dataset has, what is the categorical features. the ordinal features. or the numeric values.
pd.set_option('max_columns', 54)

X_train.describe()
# check out how much correlation there is between numeric values and the result (logerror)
# first of all, find out which columns are numeric values 

numeric_columns = X_train.select_dtypes(include=[np.number]).columns[1:-1]
numeric_columns
# keep the p-value of each columns and find out which column has good result
p_value = {}

for columns in numeric_columns:
    stats_result = regression_stats(X_train[columns], X_train['logerror']).get('p_value')
    p_value[columns] = stats_result
# sorting the p_value result of each columns
sorted_p_value = sorted(p_value.items(), key=operator.itemgetter(1))

# plot the graph with columns which has the highest p_value in the columns
for item in sorted_p_value[:5]:
    sns.jointplot(X_train[item[0]], X_train['logerror'], kind='reg')
# gather the columns which has no correlation between the column and the logerror and delete the columns later for better modeling

deleting_cols = [col for col, value in p_value.items() if value > 0.05]

deleting_cols
X_train.shape, X_test.shape
# fill a few missing value first  to see the correlationship between the categorical columns and the logerror

for dataset in combine:
    dataset['hashottuborspa'].fillna('False', inplace=True)
    dataset['propertycountylandusecode'].fillna('0100', inplace=True)
    dataset['propertyzoningdesc'].fillna('LAR1', inplace=True)
    dataset['taxdelinquencyflag'].fillna('N', inplace=True)
    dataset['fireplaceflag'].fillna('False', inplace=True)
true_std = X_train[X_train.hashottuborspa==True].logerror.std()
true_mean = X_train[X_train.hashottuborspa==True].logerror.mean()
false_std = X_train[X_train.hashottuborspa=='False'].logerror.std()
false_mean = X_train[X_train.hashottuborspa=='False'].logerror.mean()

grid = sns.FacetGrid(X_train, col='hashottuborspa')
grid_map = grid.map(sns.kdeplot, 'logerror', shade=True)
axes = grid_map.axes
axes[0, 0].set_xlim(-0.5, 0.5)
axes[0, 1].set_xlim(-0.5, 0.5)
print(f'true_std: {true_std}, true_mean: {true_mean} \nfalse_std: {false_std}, false_mean: {false_mean}')
true_std = X_train[X_train.taxdelinquencyflag=='Y'].logerror.std()
true_mean = X_train[X_train.taxdelinquencyflag=='Y'].logerror.mean()
false_std = X_train[X_train.taxdelinquencyflag=='N'].logerror.std()
false_mean = X_train[X_train.taxdelinquencyflag=='N'].logerror.mean()


grid = sns.FacetGrid(X_train, col='taxdelinquencyflag')
grid_map = grid.map(sns.kdeplot, 'logerror', shade=True)
axes = grid_map.axes
axes[0, 0].set_xlim(-0.5, 0.5)
axes[0, 1].set_xlim(-0.5, 0.5)
print(f'true_std: {true_std}, true_mean: {true_mean} \nfalse_std: {false_std}, false_mean: {false_mean}')
true_std = X_train[X_train.fireplaceflag==True].logerror.std()
true_mean = X_train[X_train.fireplaceflag==True].logerror.mean()
false_std = X_train[X_train.fireplaceflag=='False'].logerror.std()
false_mean = X_train[X_train.fireplaceflag=='False'].logerror.mean()


grid = sns.FacetGrid(X_train, col='fireplaceflag')
grid_map = grid.map(sns.kdeplot, 'logerror', shade=True)
axes = grid_map.axes
axes[0, 0].set_xlim(-0.5, 0.5)
axes[0, 1].set_xlim(-0.5, 0.5)
print(f'true_std: {true_std}, true_mean: {true_mean} \nfalse_std: {false_std}, false_mean: {false_mean}')
deleting_cols
# first of all, delete the columns which is proved not to be correlated with the logerror

for dataset in combine:
    dataset = dataset.drop(deleting_cols, axis=1)
# check how many Nan values in X_train, X_test

train_columns_values_cnt = X_train.isnull().sum(axis=0).reset_index()
train_columns_values_cnt.columns = ['col_name', 'nan_count']
train_columns_values_cnt['nan_ratio'] = train_columns_values_cnt['nan_count'] / X_train.shape[0]
train_columns_values_cnt = train_columns_values_cnt.sort_values('nan_count', ascending=False)
test_columns_values_cnt = X_test.isnull().sum(axis=0).reset_index()
test_columns_values_cnt.columns = ['col_name', 'nan_count']
test_columns_values_cnt['nan_ratio'] = test_columns_values_cnt['nan_count'] / X_test.shape[0]
test_columns_values_cnt = test_columns_values_cnt.sort_values('nan_count', ascending=False)
# delete the columns of which the missing values ratio is more than 0.6
train_deleting_col = train_columns_values_cnt.loc[train_columns_values_cnt.nan_ratio > 0.6]['col_name'].values
test_deleting_col = test_columns_values_cnt.loc[test_columns_values_cnt.nan_ratio > 0.6]['col_name'].values

X_train = X_train.drop(train_deleting_col, axis=1)
X_test = X_test.drop(test_deleting_col, axis=1)
# check how many Nan values in X_train, X_test

train_columns_values_cnt = X_train.isnull().sum(axis=0).reset_index()
train_columns_values_cnt.columns = ['col_name', 'nan_count']
train_columns_values_cnt['nan_ratio'] = train_columns_values_cnt['nan_count'] / X_train.shape[0]
train_columns_values_cnt = train_columns_values_cnt.sort_values('nan_count', ascending=False)
test_columns_values_cnt = X_test.isnull().sum(axis=0).reset_index()
test_columns_values_cnt.columns = ['col_name', 'nan_count']
test_columns_values_cnt['nan_ratio'] = test_columns_values_cnt['nan_count'] / X_test.shape[0]
test_columns_values_cnt = test_columns_values_cnt.sort_values('nan_count', ascending=False)
fig, ax = plt.subplots(ncols=2, figsize=(30, 20))
ax[0].set_title("train data nan count")
ax[1].set_title("test data nan count")
train_columns_values_cnt = train_columns_values_cnt.loc[train_columns_values_cnt.nan_ratio > 0]
test_columns_values_cnt = test_columns_values_cnt.loc[test_columns_values_cnt.nan_ratio > 0]
sns.barplot(train_columns_values_cnt.nan_ratio, train_columns_values_cnt.col_name, ax=ax[0])
sns.barplot(test_columns_values_cnt.nan_ratio, test_columns_values_cnt.col_name, ax=ax[1])
# the columns which have not that high Nan ratio will filled with using scikit-learn imputer

train_imputing_col = train_columns_values_cnt.sort_values('nan_ratio', ascending=False)[3:].col_name.values
test_imputing_col = test_columns_values_cnt.sort_values('nan_ratio', ascending=False)[3:].col_name.values

train_imputing_col, test_imputing_col

for col in train_imputing_col:
    if X_train[col].dtype == np.number:
        X_train[col].fillna(X_train[col].median(), inplace=True)
    else:
        X_train[col].fillna(X_train[col].mode()[0], inplace=True)
        
for col in test_imputing_col:
    if X_test[col].dtype == np.number:
        X_test[col].fillna(X_test[col].median(), inplace=True)
    else:
        X_test[col].fillna(X_test[col].mode()[0], inplace=True)
# check how many Nan values in X_train, X_test

train_columns_values_cnt = X_train.isnull().sum(axis=0).reset_index()
train_columns_values_cnt.columns = ['col_name', 'nan_count']
train_columns_values_cnt['nan_ratio'] = train_columns_values_cnt['nan_count'] / X_train.shape[0]
train_columns_values_cnt = train_columns_values_cnt.sort_values('nan_count', ascending=False)
test_columns_values_cnt = X_test.isnull().sum(axis=0).reset_index()
test_columns_values_cnt.columns = ['col_name', 'nan_count']
test_columns_values_cnt['nan_ratio'] = test_columns_values_cnt['nan_count'] / X_test.shape[0]
test_columns_values_cnt = test_columns_values_cnt.sort_values('nan_count', ascending=False)
fig, ax = plt.subplots(ncols=2, figsize=(30, 20))
ax[0].set_title("train data nan count")
ax[1].set_title("test data nan count")
train_columns_values_cnt = train_columns_values_cnt.loc[train_columns_values_cnt.nan_ratio > 0]
test_columns_values_cnt = test_columns_values_cnt.loc[test_columns_values_cnt.nan_ratio > 0]
sns.barplot(train_columns_values_cnt.nan_ratio, train_columns_values_cnt.col_name, estimator=np.sum, ax=ax[0])
sns.barplot(test_columns_values_cnt.nan_ratio, test_columns_values_cnt.col_name, estimator=np.sum, ax=ax[1])
# change the categorical values to ordinal values (my functions didnt't work properly. I didn't know why. I just did manually....)

X_train['hashottuborspa'] = X_train['hashottuborspa'].astype('category').cat.codes
X_train['propertycountylandusecode'] = X_train['propertycountylandusecode'].astype('category').cat.codes
X_train['propertyzoningdesc'] = X_train['propertyzoningdesc'].astype('category').cat.codes
X_train['fireplaceflag'] = X_train['fireplaceflag'].astype('category').cat.codes
X_train['taxdelinquencyflag'] = X_train['taxdelinquencyflag'].astype('category').cat.codes

X_test['hashottuborspa'] = X_test['hashottuborspa'].astype('category').cat.codes
X_test['propertycountylandusecode'] = X_test['propertycountylandusecode'].astype('category').cat.codes
X_test['propertyzoningdesc'] = X_test['propertyzoningdesc'].astype('category').cat.codes
X_test['fireplaceflag'] = X_test['fireplaceflag'].astype('category').cat.codes
X_test['taxdelinquencyflag'] = X_test['taxdelinquencyflag'].astype('category').cat.codes
# In terms of nan values ratio, top 3 columns will be imputed by predicting Random Forest

imputing_col = train_columns_values_cnt.sort_values('nan_ratio', ascending=False)[:3].col_name.values
imputing_col
# make dataset for using imputing

imp_dataset = pd.concat([X_train, X_test])
imp_dataset = imp_dataset.drop('logerror', axis=1)
imp_dataset.shape
# can't fit all data from the dataset becasue of the performance of my computer. So, just choose 10 percent of the dataset randomly for fitting and predicting

impute_train_dataset = shuffle(imp_dataset.loc[~imp_dataset.heatingorsystemtypeid.isnull()], random_state=0)[:180640]
# column heatingorsystemtypeid prediction preparation

impute_X_train = impute_train_dataset.drop(imputing_col, axis=1)
impute_y_train = impute_train_dataset['heatingorsystemtypeid'].values
impute_X_test = imp_dataset.loc[imp_dataset.heatingorsystemtypeid.isnull()].drop(imputing_col, axis=1)
# Random Forest accuracy

tuned_parameters = [
    {'criterion': ['gini', 'entropy']}
]

random_forest = RandomForestClassifier(n_estimators=100)

clf = GridSearchCV(random_forest, tuned_parameters, cv=2)
clf.fit(impute_X_train, impute_y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
random_forest_pred = clf.predict(impute_X_test)
acc_random_forest = round(clf.score(impute_X_train, impute_y_train) * 100, 2)
print()
print(acc_random_forest)
# column heatingorsystemtypeid prediction result

heatingorsystemtypeid_prediction = pd.DataFrame(random_forest_pred, index=impute_X_test['parcelid']).reset_index()
heatingorsystemtypeid_prediction.shape
# column buildingqualitytypeid prediction preparation

impute_train_dataset = shuffle(imp_dataset.loc[~imp_dataset.buildingqualitytypeid.isnull()], random_state=0)[:180640]
impute_X_train = impute_train_dataset.drop(imputing_col, axis=1)
impute_y_train = impute_train_dataset['buildingqualitytypeid'].values
impute_X_test = imp_dataset.loc[imp_dataset.buildingqualitytypeid.isnull()].drop(imputing_col, axis=1)
# Random Forest accuracy

tuned_parameters = [
    {'criterion': ['gini', 'entropy']}
]

random_forest = RandomForestClassifier(n_estimators=100)

clf = GridSearchCV(random_forest, tuned_parameters, cv=2)
clf.fit(impute_X_train, impute_y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
random_forest_pred = clf.predict(impute_X_test)
acc_random_forest = round(clf.score(impute_X_train, impute_y_train) * 100, 2)
print()
print(acc_random_forest)
# column buildingqualitytypeid prediction result

buildingqualitytypeid_prediction = pd.DataFrame(random_forest_pred, index=impute_X_test['parcelid']).reset_index()
buildingqualitytypeid_prediction.shape
# column unitcnt prediction preparation

impute_train_dataset = shuffle(imp_dataset.loc[~imp_dataset.unitcnt.isnull()], random_state=0)[:180640]
impute_X_train = impute_train_dataset.drop(imputing_col, axis=1)
impute_y_train = impute_train_dataset['unitcnt'].values
impute_X_test = imp_dataset.loc[imp_dataset.unitcnt.isnull()].drop(imputing_col, axis=1)
# Random Forest accuracy

tuned_parameters = [
    {'criterion': ['gini', 'entropy']}
]

random_forest = RandomForestClassifier(n_estimators=100)

clf = GridSearchCV(random_forest, tuned_parameters, cv=2)
clf.fit(impute_X_train, impute_y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
random_forest_pred = clf.predict(impute_X_test)
acc_random_forest = round(clf.score(impute_X_train, impute_y_train) * 100, 2)
print()
print(acc_random_forest)
# column unitcnt prediction result

unitcnt_prediction = pd.DataFrame(random_forest_pred, index=impute_X_test['parcelid']).reset_index()
unitcnt_prediction.shape
# change the columns name to make the task easier laster

heatingorsystemtypeid_prediction.columns = ['parcelid', 'heatingorsystemtypeid']
buildingqualitytypeid_prediction.columns = ['parcelid', 'buildingqualitytypeid']
unitcnt_prediction.columns = ['parcelid', 'unitcnt']
# lastly, fill out all missing values

X_train.loc[X_train.heatingorsystemtypeid.isnull(), 'heatingorsystemtypeid'] = \
    heatingorsystemtypeid_prediction.loc[heatingorsystemtypeid_prediction.parcelid.isin(X_train.loc[X_train.heatingorsystemtypeid.isnull()]['parcelid'].values)]['heatingorsystemtypeid'].values
X_test.loc[X_test.heatingorsystemtypeid.isnull(), 'heatingorsystemtypeid'] = \
    heatingorsystemtypeid_prediction.loc[heatingorsystemtypeid_prediction.parcelid.isin(X_test.loc[X_test.heatingorsystemtypeid.isnull()]['parcelid'].values)]['heatingorsystemtypeid'].values
X_train.loc[X_train.buildingqualitytypeid.isnull(), 'buildingqualitytypeid'] = \
    buildingqualitytypeid_prediction.loc[buildingqualitytypeid_prediction.parcelid.isin(X_train.loc[X_train.buildingqualitytypeid.isnull()]['parcelid'].values)]['buildingqualitytypeid'].values
X_test.loc[X_test.buildingqualitytypeid.isnull(), 'buildingqualitytypeid'] = \
    buildingqualitytypeid_prediction.loc[buildingqualitytypeid_prediction.parcelid.isin(X_test.loc[X_test.buildingqualitytypeid.isnull()]['parcelid'].values)]['buildingqualitytypeid'].values
X_train.loc[X_train.unitcnt.isnull(), 'unitcnt'] = \
    unitcnt_prediction.loc[unitcnt_prediction.parcelid.isin(X_train.loc[X_train.unitcnt.isnull()]['parcelid'].values)]['unitcnt'].values
X_test.loc[X_test.unitcnt.isnull(), 'unitcnt'] = \
    unitcnt_prediction.loc[unitcnt_prediction.parcelid.isin(X_test.loc[X_test.unitcnt.isnull()]['parcelid'].values)]['unitcnt'].values
last_test_dataset = pd.concat([X_train, X_test]).drop('logerror', axis=1)
last_test_dataset.shape
# save the final dataset for the unexpected situation

y_train_csv = pd.DataFrame(data=y_train, index=X_train.parcelid.values)
y_train_csv.columns = ['logerror']

X_train.to_csv('x_train.csv')
y_train_csv.to_csv('y_train.csv')
last_test_dataset.to_csv('x_test.csv')
# change the dtypes of dataframes for using less memory

for c, dtype in zip(X_train.columns, X_train.dtypes):
    if dtype == np.float64:	
        X_train[c] = X_train[c].astype(np.float32)
        
for c, dtype in zip(last_test_dataset.columns, last_test_dataset.dtypes):
    if dtype == np.float64:	
        last_test_dataset[c] = last_test_dataset[c].astype(np.float32)

X_train = X_train.drop('logerror', axis=1)
last_test_dataset = last_test_dataset[X_train.columns]
# using ightgbm algorithm and predict the logerror or each properties

train = lgb.Dataset(X_train, label=y_train)

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          
params['sub_feature'] = 0.5      
params['bagging_fraction'] = 0.85 
params['bagging_freq'] = 40
params['num_leaves'] = 512 
params['min_data'] = 500
params['min_hessian'] = 0.05
params['verbose'] = 0

clf = lgb.train(params, train, 430)

# to decrease the memory loss
del train; gc.collect()

clf.reset_parameter({"num_threads":1})
lgb_pred = clf.predict(last_test_dataset)
lgb_pred
# using xgboost algorithm and predict the logerror or each properties

y_mean = np.mean(y_train)

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

train = xgb.DMatrix(X_train, y_train)
test = xgb.DMatrix(last_test_dataset)

cv_result = xgb.cv(
    xgb_params, 
    dtrain, 
    nfold=5,
    num_boost_round=350,
    early_stopping_rounds=50,
    verbose_eval=10, 
    show_stdv=False,
)
num_boost_rounds = len(cv_result)

model = xgb.train(dict(xgb_params, silent=1), train, num_boost_round=num_boost_rounds)
xgb_pred = model.predict(test)
xgb_pred
submission_sample = pd.read_csv('../input/sample_submission.csv')
submission = submission_sample.sort_values(by='ParcelId')
submission.head()
xgb_submission = pd.DataFrame({'ParcelId': submission.ParcelId.astype(np.int32),
    '201610': xgb_pred, '201611': xgb_pred, '201612': xgb_pred,
    '201710': xgb_pred, '201711': xgb_pred, '201712': xgb_pred
})
xgb_submission.head()
lgb_submission = pd.DataFrame({'ParcelId': submission.ParcelId.astype(np.int32),
    '201610': lgb_pred, '201611': lgb_pred, '201612': lgb_pred,
    '201710': lgb_pred, '201711': lgb_pred, '201712': lgb_pred
})
lgb_submission.head()
xgb_submission = xgb_submission[['ParcelId', '201610', '201611', '201612', '201710', '201711', '201712']]
lgb_submission = lgb_submission[['ParcelId', '201610', '201611', '201612', '201710', '201711', '201712']]
# export the result for submitting
xgb_submission.to_csv('output.csv', float_format='%.4g', index=False)
lgb_submission.to_csv('output_lgb.csv', float_format='%.4g', index=False)