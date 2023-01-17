import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import os

import seaborn as sns

from sklearn import model_selection, preprocessing

import xgboost as xgb

color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option('display.max_columns', 500)

from zipfile import ZipFile 
os.listdir('../input/sberbank-russian-housing-market')
train_df = pd.read_csv(ZipFile("../input/sberbank-russian-housing-market/train.csv.zip").open('train.csv'), parse_dates=['timestamp'])

test_df = pd.read_csv(ZipFile("../input/sberbank-russian-housing-market/test.csv.zip").open('test.csv'), parse_dates=['timestamp'])

macro_df = pd.read_csv(ZipFile("../input/sberbank-russian-housing-market/macro.csv.zip").open('macro.csv'), parse_dates=['timestamp'])
train_df.head()
macro_df.head()
train_df = pd.merge(train_df, macro_df, how="left", on="timestamp")

test_df = pd.merge(test_df, macro_df, how="left", on="timestamp")
train_df.shape, test_df.shape
train_df['price_doc'].dtype
plt.figure(figsize=(8,6))

plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('price', fontsize=12)

plt.show()
plt.figure(figsize=(12,8))

sns.distplot(train_df.price_doc.values, bins=50, kde=True)

plt.xlabel('price')

plt.show()
plt.figure(figsize=(12,8))

sns.distplot(np.log(train_df.price_doc.values), bins=50, kde=True)

plt.xlabel('price')

plt.show()
upper_limit = np.percentile(train_df['price_doc'], 99)

lower_limit = np.percentile(train_df['price_doc'], 1)



train_df.loc[(train_df['price_doc'] > upper_limit), 'price_doc'] = upper_limit

train_df.loc[(train_df['price_doc'] < lower_limit), 'price_doc'] = lower_limit
missing_df = train_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.loc[(missing_df['missing_count'] > 0), :]

missing_df = missing_df.sort_values(by='missing_count')

ind = range(missing_df.shape[0])



fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missing_df['missing_count'], color="purple")

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
cat_cols = [col for col in train_df.columns if train_df[col].dtype == 'object']



le = preprocessing.LabelEncoder()



for col in cat_cols:

    train_df[col] = le.fit_transform(train_df[col].astype('str'))

    test_df[col] = le.fit_transform(test_df[col].astype('str'))
plt.figure(figsize=(12,8))

sns.countplot(x='floor', data=train_df)

plt.ylabel('Count', fontsize=12)

plt.xlabel('floor number', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
temp_df = train_df.groupby(['floor'])['price_doc'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))

sns.pointplot(x='floor', y='price_doc', data=temp_df)

plt.ylabel('Median Price', fontsize=12)

plt.xlabel('Floor number', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(x="max_floor", data=train_df)

plt.ylabel('Count', fontsize=12)

plt.xlabel('Max floor number', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(x="max_floor", y="price_doc", data=train_df)

plt.ylabel('Median Price', fontsize=12)

plt.xlabel('Max Floor number', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
# count null values of each row

train_df['null_count'] = train_df.isnull().sum(axis=1)

test_df['null_count'] = test_df.isnull().sum(axis=1)



# plot to check affect of null values on the pric_doc col, point plot shows only the mean (or other estimator) value,

plt.figure(figsize=(20, 8))

sns.pointplot(x='null_count', y='price_doc', data=train_df)

plt.ylabel('price_doc', fontsize=12)

plt.xlabel('null_count', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
train_df.fillna(-99, inplace=True)

test_df.fillna(-99, inplace=True)
# year and month

train_df['yearmonth'] = train_df['timestamp'].dt.year*100 + train_df['timestamp'].dt.month

test_df['yearmonth'] = test_df['timestamp'].dt.year*100 + test_df['timestamp'].dt.month



# year and week

train_df['yearweek'] = train_df['timestamp'].dt.year*100 + train_df['timestamp'].dt.weekofyear

test_df['yearweek'] = test_df['timestamp'].dt.year*100 + test_df['timestamp'].dt.weekofyear



# year

train_df['year'] = train_df['timestamp'].dt.year

test_df['year'] = test_df['timestamp'].dt.year



# month of year

train_df['month_of_year'] = train_df['timestamp'].dt.month

test_df['month_of_year'] = test_df['timestamp'].dt.month



# week of year 

train_df["week_of_year"] = train_df["timestamp"].dt.weekofyear

test_df["week_of_year"] = test_df["timestamp"].dt.weekofyear



# day of week 

train_df["day_of_week"] = train_df["timestamp"].dt.weekday

test_df["day_of_week"] = test_df["timestamp"].dt.weekday
plt.figure(figsize=(12,8))

sns.pointplot(x='yearweek', y='price_doc', data=train_df)

plt.ylabel('price_doc', fontsize=12)

plt.xlabel('yearweek', fontsize=12)

plt.title('Median Price distribution by year and week_num')

plt.xticks(rotation='vertical')

plt.show()
plt.figure(figsize=(12,8))

sns.pointplot(x='week_of_year', y='price_doc', data=train_df)

plt.ylabel('price_doc', fontsize=12)

plt.xlabel('week_of_year', fontsize=12)

plt.title('Median Price distribution by week of year')

plt.xticks(rotation='vertical')

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(x='month_of_year', y='price_doc', data=train_df)

plt.ylabel('price_doc', fontsize=12)

plt.xlabel('month_of_year', fontsize=12)

plt.title('Median Price distribution by month_of_year')

plt.xticks(rotation='vertical')

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(x='day_of_week', y='price_doc', data=train_df)

plt.ylabel('price_doc', fontsize=12)

plt.xlabel('day_of_week', fontsize=12)

plt.title('Median Price distribution by day of week')

plt.xticks(rotation='vertical')

plt.show()
# ratio of living area to full area

train_df["ratio_life_sq_full_sq"] = train_df["life_sq"] / np.maximum(train_df["full_sq"].astype("float"), 1)

test_df["ratio_life_sq_full_sq"] = test_df["life_sq"] / np.maximum(test_df["full_sq"].astype("float"),1)



train_df.loc[(train_df["ratio_life_sq_full_sq"] < 0), "ratio_life_sq_full_sq"] = 0

train_df.loc[(train_df["ratio_life_sq_full_sq"] > 1), "ratio_life_sq_full_sq"] = 1



test_df.loc[(test_df["ratio_life_sq_full_sq"] < 0), "ratio_life_sq_full_sq"] = 0

test_df.loc[(test_df["ratio_life_sq_full_sq"] > 1), "ratio_life_sq_full_sq"] = 1



# ratio of kitchen area to living area 

train_df["ratio_kitch_sq_life_sq"] = train_df["kitch_sq"] / np.maximum(train_df["life_sq"].astype("float"),1)

test_df["ratio_kitch_sq_life_sq"] = test_df["kitch_sq"] / np.maximum(test_df["life_sq"].astype("float"),1)



train_df.loc[(train_df["ratio_kitch_sq_life_sq"] < 0), "ratio_kitch_sq_life_sq"] = 0

train_df.loc[(train_df["ratio_kitch_sq_life_sq"] > 1), "ratio_kitch_sq_life_sq"] = 1



test_df.loc[(test_df["ratio_kitch_sq_life_sq"] < 0), "ratio_kitch_sq_life_sq"] = 0

test_df.loc[(test_df["ratio_kitch_sq_life_sq"] > 1), "ratio_kitch_sq_life_sq"] = 1



# ratio of kitchen area to full area #

train_df["ratio_kitch_sq_full_sq"] = train_df["kitch_sq"] / np.maximum(train_df["full_sq"].astype("float"),1)

test_df["ratio_kitch_sq_full_sq"] = test_df["kitch_sq"] / np.maximum(test_df["full_sq"].astype("float"),1)



train_df.loc[(train_df["ratio_kitch_sq_full_sq"] < 0), "ratio_kitch_sq_full_sq"] = 0

train_df.loc[(train_df["ratio_kitch_sq_full_sq"] > 1), "ratio_kitch_sq_full_sq"] = 1



test_df.loc[(test_df["ratio_kitch_sq_full_sq"] < 0), "ratio_kitch_sq_full_sq"] = 0

test_df.loc[(test_df["ratio_kitch_sq_full_sq"] > 1), "ratio_kitch_sq_full_sq"] = 1
plt.figure(figsize=(12,12))

sns.jointplot(x=train_df["ratio_life_sq_full_sq"], y=np.log1p(train_df["price_doc"]), size=10)

plt.ylabel('Log of Price', fontsize=12)

plt.xlabel('Ratio of living area to full area', fontsize=12)

plt.title("Joint plot on log of living price to ratio_life_sq_full_sq")

plt.show()
plt.figure(figsize=(12,12))

sns.jointplot(x=train_df["ratio_life_sq_full_sq"], y=np.log1p(train_df["price_doc"]), kind='kde',size=10)

plt.ylabel('Log of Price', fontsize=12)

plt.xlabel('Ratio of kitchen area to living area', fontsize=12)

plt.title("Joint plot on log of living price to ratio_kitch_sq_life_sq")

plt.show()
# floor of the house to the total number of floors in the house 

train_df["ratio_floor_max_floor"] = train_df["floor"] / train_df["max_floor"].astype("float")

test_df["ratio_floor_max_floor"] = test_df["floor"] / test_df["max_floor"].astype("float")



# num of floor from top

train_df["floor_from_top"] = train_df["max_floor"] - train_df["floor"]

test_df["floor_from_top"] = test_df["max_floor"] - test_df["floor"]



# difference between full area and living area

train_df["extra_sq"] = train_df["full_sq"] - train_df["life_sq"]

test_df["extra_sq"] = test_df["full_sq"] - test_df["life_sq"]



# age of the building

train_df["age_of_building"] = train_df["build_year"] - train_df["year"]

test_df["age_of_building"] = test_df["build_year"] - test_df["year"]



# effect of school

train_df["ratio_preschool"] = train_df["children_preschool"] / train_df["preschool_quota"].astype("float")

test_df["ratio_preschool"] = test_df["children_preschool"] / test_df["preschool_quota"].astype("float")



train_df["ratio_school"] = train_df["children_school"] / train_df["school_quota"].astype("float")

test_df["ratio_school"] = test_df["children_school"] / test_df["school_quota"].astype("float")
def count_by_dates(df, col):

    temp_df = df.groupby(col)["id"].aggregate("count").reset_index()

    temp_df.columns = [col, "count_" + col]

    df = pd.merge(df, temp_df, on=col, how="left")

    return df



train_df = count_by_dates(train_df, "yearmonth")

test_df = count_by_dates(test_df, "yearmonth")



train_df = count_by_dates(train_df, "yearweek")

test_df = count_by_dates(test_df, "yearweek")
train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

train_y = np.log1p(train_df["price_doc"])



test_X = test_df.drop(["id", "timestamp"] , axis=1)
val_time = 201407



dev_X = train_X[(train_X["yearmonth"] < val_time)]

dev_y = train_y[(train_X["yearmonth"] < val_time)]



val_X = train_X[(train_X["yearmonth"] >= val_time)]

val_y = train_y[(train_X["yearmonth"] >= val_time)]
print(dev_X.shape, dev_y.shape)

print(val_X.shape, val_y.shape)
xgb_params = {

    'eta': 0.05,

    'max_depth': 4,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'min_child_weight':1,

    'silent': 1,

    'seed':0

}

num_rounds = 100



xgtrain = xgb.DMatrix(dev_X, dev_y, feature_names = dev_X.columns)

xgtest = xgb.DMatrix(val_X, val_y, feature_names = val_X.columns)

watchlist = [(xgtrain, 'train'), (xgtest, 'test')]

model = xgb.train(xgb_params, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=5)
fig, ax = plt.subplots(figsize=(12, 18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()