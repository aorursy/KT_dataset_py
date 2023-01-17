# function for finding categorical features

def categ_props(data):
    columns_to_del = []
    for c in data.columns:
        try:
            float(data[c].values[1])
        except:
            columns_to_del.append(c)
    return columns_to_del
from feature_selector import FeatureSelector
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

%pylab inline
#import DATA

x = pd.read_csv('../input/train.csv')
x.timestamp = pd.to_datetime(x.timestamp)

x_test = pd.read_csv('../input/test.csv')
x_test.timestamp = pd.to_datetime(x_test.timestamp)

x_macro = pd.read_csv('../input/macro.csv')
x_macro.timestamp = pd.to_datetime(x_macro.timestamp)

print('x:{0} x_test:{1} x_macro:{2}'.format(x.shape, x_test.shape, x_macro.shape))
# Concatenate train + test + macro

test_ID = x_test.id
y_all = np.log1p(x["price_doc"])
x.price_doc = y_all

x_length = x.shape[0]
x_all = pd.concat([x, x_test])
x_all = pd.merge_ordered(x_all, x_macro, on='timestamp', how='left')
# Check for features with too math missing data

fs = FeatureSelector(x_all, x_all.columns)
fs.identify_missing(0.7)
fs.missing_stats.head()
# Check for features with only one value

fs.identify_single_unique()
fs.unique_stats.head()
# Remove bad features

x_all.drop(['id',                                 # unnesesary
            'provision_retail_space_modern_sqm',  # too match missing values
            'provision_retail_space_sqm',         # too match missing values
            'child_on_acc_pre_school',            # bad values 
            'modern_education_share',             # bad values
            'old_education_build_share'           # bad values
           ], axis=1, inplace=True)
# Process categorical features

cat_props = categ_props(x_all)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

x_all.product_type.fillna('noType', inplace=True)

for cat_col in cat_props:
    print(cat_col)
    x_all[cat_col]=le.fit_transform(x_all[cat_col]) 
# Create several time-based features

x_all['d'] = x.timestamp.dt.year+x.timestamp.dt.dayofyear/365
x_all['w'] = x.timestamp.dt.year+x.timestamp.dt.weekofyear/48
x_all['m'] = x.timestamp.dt.year+x.timestamp.dt.month/12

x_all['wofy'] = x.timestamp.dt.weekofyear
x_all['mofy'] = x.timestamp.dt.month
x_all['dofw'] = x.timestamp.dt.dayofweek

# Create relative features
x_all['rel_kitch_sq'] = x_all['kitch_sq'] / x_all['full_sq'].astype(float)
x_all['rel_life_sq'] = x_all['life_sq'] / x_all['full_sq'].astype(float)
x_all.rel_kitch_sq[x_all.rel_kitch_sq>1]=1
x_all.rel_life_sq[x_all.rel_life_sq>1]=1

x_all['rel_floor'] = x_all['floor'] / x_all['max_floor'].astype(float)
x_all.rel_floor[x_all.rel_floor>1]=1

x_all['brent_rub'] = x_all['brent'] * x_all['usdrub'].astype(float)

x_all.drop(['timestamp'], axis=1, inplace=True)    
    
print(x_all.shape)
# Create a validation set, with last 20% of data

x_all.replace([np.inf, -np.inf], np.nan)
x_all.fillna(0, inplace=True)

num_val = int(x_length * 0.2)

x_train_all = x_all[:x_length]
x_train = x_all[:x_length-num_val]
x_val = x_all[x_length-num_val:x_length]
y_train = y_all[:x_length-num_val]
y_val = y_all[x_length-num_val:x_length]

x_test = x_all[x_length:]

print('x_train:{0} x_val:{1} x_test:{2}'.format(x_train.shape, x_val.shape, x_test.shape))
# Time averaged features

pmy = x_train.groupby('mofy')['price_doc'].aggregate(np.mean)
x_train['p_mofy'] = x_train.mofy.map(pmy)
x_val['p_mofy'] = x_val.mofy.map(pmy)
x_test['p_mofy'] = x_test.mofy.map(pmy)

pwy = x_train.groupby('wofy')['price_doc'].aggregate(np.mean)
x_train['p_wofy'] = x_train.wofy.map(pwy)
x_val['p_wofy'] = x_val.wofy.map(pwy)
x_test['p_wofy'] = x_test.wofy.map(pwy)

pbw = x_train.groupby('dofw')['price_doc'].aggregate(np.mean)
x_train['p_pdw'] = x_train.dofw.map(pbw)
x_val['p_pdw'] = x_val.dofw.map(pbw)
x_test['p_pdw'] = x_test.dofw.map(pbw)

fig, ax = plt.subplots(1, 1, figsize=(17, 5), sharey=True)
plt.subplot(131)
plt.xlabel('month')
plt.ylabel('average price')
plt.plot(pmy, '.-');

plt.subplot(132)
plt.xlabel('week of year')
plt.plot(pwy, '.-');

plt.subplot(133)
plt.xlabel('day of week')
plt.plot(pbw, '.-');
# Distribution of prices
# There are two points with unrealistic buil year and significant amount of zero values

i_del = x_train[x_train.build_year>2018].index
fig, ax = plt.subplots(1, 3, figsize=(17, 3))

plt.subplot(131)
plt.title('train data')
plt.xlabel('build year')
plt.ylabel('amount of points')
plt.hist(x_train.drop(i_del).build_year, bins=100);
plt.subplot(132)
plt.title('validate data')
plt.xlabel('build year')
plt.hist(x_val.build_year, bins=100);
plt.subplot(133)
plt.title('test data')
plt.xlabel('build year')
plt.hist(x_test.build_year, bins=100);
# Distribution of prices (for build year > 1900)

fig, ax = plt.subplots(1, 3, figsize=(17, 3))
plt.subplot(131)
plt.title('train data')
plt.xlabel('build year')
plt.ylabel('amount of points')
plt.hist(x_train.drop(i_del).build_year[x_train.build_year>1900], bins=100);
plt.subplot(132)
plt.title('validate data')
plt.xlabel('build year')
plt.hist(x_val.build_year[x_val.build_year>1900], bins=100);
plt.subplot(133)
plt.title('test data')
plt.xlabel('build year')
plt.hist(x_test.build_year[x_test.build_year>1900], bins=100);
buid_y_means = x_train.drop(i_del).groupby('build_year')['price_doc'].aggregate(np.mean)
fig, ax = plt.subplots(1, 1, figsize=(17, 6))
plt.xlabel('build year')
plt.ylabel('average price')
plt.xlim([1850,2018])
plt.bar(x = buid_y_means.index, height = buid_y_means-13, bottom = 13);
# Average price - time dependence. There is notisable linear correlation
# Unfortunately this approximation can't improve regression task significantly

pm_train = x_train.groupby('m')['price_doc'].aggregate(np.mean)
pm_val = x_val.groupby('m')['price_doc'].aggregate(np.mean)
p = np.polyfit(pm_train.index, pm_train.values, 1)

fig, ax = plt.subplots(1, 1, figsize=(17, 6))
plt.xlabel('time')
plt.ylabel('average price')
plt.plot(x_train.groupby('m')['price_doc'].aggregate(np.mean));
plt.plot(x_val.groupby('m')['price_doc'].aggregate(np.mean));
plt.plot(pm_train.index, np.polyval(p, pm_train.index));
plt.plot(pm_val.index, np.polyval(p, pm_val.index),'--');
# Delete target
x_train.drop("price_doc", axis = 1, inplace = True)
x_val.drop("price_doc", axis = 1, inplace = True)
x_test.drop("price_doc", axis = 1, inplace = True)
print('x_train:{0} x_val:{1} x_test:{2}'.format(x_train.shape, x_val.shape, x_test.shape))
# Find collinear features (correlation > 0.99).
# There are many collinear features in particular dataset such metro_km_avto and metro_min_avto. 
fs = FeatureSelector(x_train, y_train)
fs.identify_collinear(0.99)
print(fs.ops['collinear'][:20])
# Identify features with zero importance
fs.identify_zero_importance(task='regression', eval_metric='rmse', n_iterations=5)
# Identify features with low importance

fs.identify_low_importance(cumulative_importance=0.99)
fs.plot_feature_importances(plot_n=20, threshold=0.99)
# Drop selected features

x_train = fs.remove('all')
x_val.drop(fs.removed_features, axis=1, inplace=True)
x_test.drop(fs.removed_features, axis=1, inplace=True)

print('x_train:{0} x_val:{1} x_test:{2}'.format(x_train.shape, x_val.shape, x_test.shape))
# Train XGBoost model and validate results

import xgboost as xgb
from sklearn import metrics
clf = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, min_child_weight=20)
clf.fit(x_train, y_train)

print(metrics.mean_squared_error(y_val, clf.predict(x_val))**0.5)
# Plot importances of XGBoost model
# Some of created features can be noticed in top 50 important features!
fig, ax = plt.subplots(1, 1, figsize=(8, 16))
xgb.plot_importance(clf, max_num_features=50, height=0.5, ax=ax);
# Plot true values vs preicted ones

plt.scatter(y_train, clf.predict(x_train), alpha=0.3, c='red')
plt.scatter(y_val, clf.predict(x_val), alpha=0.3, c='blue');
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.axis([13,19,13,19])
plt.plot([13,19],[13,19]);
# Train model on all data
clf.fit(pd.concat([x_train, x_val]), y_all)
y_sub = clf.predict(x_test)
submit_df = pd.DataFrame({'id':test_ID, 'price_doc':np.expm1(y_sub)})
submit_df.to_csv('submit_data.csv',index=False)