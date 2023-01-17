import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

from sklearn import model_selection, preprocessing

import xgboost as xgb



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("train:", train.shape)

print("test:" ,test.shape)
train['price_doc'].describe()
plt.figure(figsize=(12,8))

sns.violinplot([train.price_doc])

plt.xlabel('price', fontsize=12)
#skewness and kurtosis

print("Skewness: %f" % train['price_doc'].skew())

print("Kurtosis: %f" % train['price_doc'].kurt())
plt.figure(figsize=(12,8))

sns.violinplot(np.log(train.price_doc.values))

plt.xlabel('price', fontsize=12)

plt.show()
train['timestamp']=pd.to_datetime(train.timestamp, format ='%Y %m %d',errors='coerce');

x=pd.DataFrame([train.timestamp.dt.month,train.timestamp.dt.year,train.price_doc]).T

x.columns=['month','year','price']

y=x.groupby(['year','month'])['price'].aggregate(np.std).reset_index()

plt.figure(figsize=(12,8))

color = sns.color_palette()

sns.barplot(y.year.astype(str).str.cat(y.month.astype(str), sep='-'), y.price.values, alpha=0.8, color=color[3])

plt.ylabel('Median Price', fontsize=15)

plt.xlabel('Year-Month', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
for f in train.columns:

    if train[f].dtype=='object':

        lbl = preprocessing.LabelEncoder()

        train[f] = lbl.fit_transform(list(train[f].values))

        

train_y = train.price_doc.values

train_X = train.drop(["id", "timestamp", "price_doc"], axis=1)



xgb_params = {

    'eta': 0.02,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}

dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)



# plot the important features #

fig, ax = plt.subplots(figsize=(6,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
#box plot overallqual/saleprice

var = 'full_sq'

data = pd.concat([train['price_doc'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(13, 6))

fig = sns.boxplot(x=var, y="price_doc", data=data)

plt.xticks(rotation=90);
#correlation matrix

train_df=train[['full_sq','life_sq','floor','build_year','max_floor','kitch_sq','num_room','state','school_km','metro_min_avto','kindergarten_km','material','railroad_km', 'swim_pool_km','sub_area','industrial_km','green_zone_km','price_doc']]

corrmat = train_df.corr()

f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(corrmat, vmax=1, square=True, annot=True);
#missing data

train_missing=train.isnull().sum(axis=0).reset_index() #return true/false for missing. sum number of misses

train_missing.columns=["column header", "missing count"] # define header

train_missing=train_missing.ix[train_missing["missing count"]>0]

plt.figure(figsize=(8,20))

sns.barplot(train_missing['missing count'],train_missing['column header'],alpha=1, color=color[5])

plt.ylabel('Header', fontsize=15)

plt.xlabel('Missing Counts', fontsize=15)

plt.show()
print('max value in build year:', np.max(train_df['build_year']))

print('min value in build year:',np.min(train_df['build_year']))

print('mean value in build year:',np.mean(train_df['build_year']))

year=pd.DataFrame(np.sort(train_df['build_year']))

year.columns=['year']

year