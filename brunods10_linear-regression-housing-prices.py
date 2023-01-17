9 + 10
# Data science.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Data cleaning

from sklearn.impute import SimpleImputer



# Data visualization.

import matplotlib.pyplot as plt

import seaborn as sns



# Machine learning.

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import f_regression

from sklearn.feature_selection import RFE
with open('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt', 'r') as f:

    descriptions = f.read()    
dict_desc = {}

for i, j in enumerate(descriptions.split('\n')):

    if j.isspace() or (len(j) == 0):

        continue

    if not j.startswith(' '):

        key_name, key_desc = j.strip().split(': ')        

        dict_desc[key_name] = [key_desc]

    else:

        dict_desc[key_name].append(j)

        dict_desc[key_name] = ['\n'.join(dict_desc[key_name])]        
dict_desc = {k:'\n'.join(v) for k, v in dict_desc.items()}
print(dict_desc.keys(), '\n')

print(dict_desc['HouseStyle'])
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col=[0])

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col=[0])
df.head()
X = df.drop('SalePrice', axis=1)

y = df['SalePrice']

X_train, X_dev, y_train, y_dev = train_test_split(X, y,

                                                 train_size=0.8,

                                                 random_state=42)
X_train.head()
sns.distplot(y_train);
sns.boxplot(data=df, x='BldgType', y='SalePrice');
sns.scatterplot(x=X_train['LotArea'], y=y_train);
sns.scatterplot(x=np.log2(X_train['LotArea'] + 1), y=y_train);
np.log(X_train['LotArea'].values)
X_train
X_train.info()
X_train.nunique()
num_cat = 25

count_unique = X_train.nunique()

cols_cat = X_train.columns[count_unique <= num_cat]

cols_cat
count_unique[cols_cat].sort_values().tail(10)
print(dict_desc['MSSubClass'])
X_train['MoSold'].value_counts()
dict_to_str = {k:'category' for k in cols_cat}  # dict comprehension

X_train = X_train.astype(dict_to_str)

X_dev = X_dev.astype(dict_to_str)
# check text features

X_train.select_dtypes(np.character)
X_train.isnull().sum().sort_values()
missing_values = X_train.isnull().sum()

pct_missing_values = missing_values / X_train.shape[0]

idx = pct_missing_values > 0.25

pct_missing_values[idx].plot(kind='bar');
missing_values = X_dev.isnull().sum()

pct_missing_values = missing_values / X_dev.shape[0]

idx = pct_missing_values > 0.25

pct_missing_values[idx].plot(kind='bar');
# remove columns with too many missing values from train and dev sets

cols_drop = X_train.columns[idx]

X_train = X_train.drop(cols_drop, axis=1)

X_dev = X_dev[X_train.columns]
cols_missing_train = X_train.columns[X_train.isnull().any()]

cols_missing_dev = X_dev.columns[X_dev.isnull().any()]

cols_missing = set(cols_missing_train).intersection(cols_missing_dev)

X_train[cols_missing].info()
X_train.select_dtypes(np.number)
'MISSING'
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

imp_median.fit(X_train.select_dtypes(np.number))

imp_median
cols_num = X_train.select_dtypes(np.number).columns

X_train[cols_num] = imp_median.transform(X_train[cols_num])

X_dev[cols_num] = imp_median.transform(X_dev[cols_num])
X_train.select_dtypes(np.number).info()
X_train.isnull().sum().sort_values()
imp_constant = SimpleImputer(missing_values=np.nan, strategy='constant')

imp_constant.fit(X_train)

imp_constant.transform(X_train)
X_train
X_train.fillna('MISSING')
cat_cols = X_train.select_dtypes('category').columns

X_train[cat_cols] = X_train[cat_cols].apply(lambda x: x.cat.add_categories('missing_value').fillna('missing_value'))

X_dev[cat_cols] = X_dev[cat_cols].apply(lambda x: x.cat.add_categories('missing_value').fillna('missing_value'))
X_train
X_dev.isnull().any().value_counts()
X_train.info()
X_dummies = pd.get_dummies(pd.concat([X_train, X_dev], axis=0))

X_train = X_dummies.reindex(X_train.index)

X_dev = X_dummies.reindex(X_dev.index)
X_train
X_dev
reg = LinearRegression()

reg.fit(X_train, y_train)
y_pred = reg.predict(X_train)

mean_squared_error(y_train, y_pred)
y_pred_dev = reg.predict(X_dev)

mean_squared_error(y_dev, y_pred_dev)
mean_squared_error(y_dev, y_pred_dev) / mean_squared_error(y_train, y_pred)
y_train_log = np.log(1 + y_train)

y_dev_log = np.log(1 + y_dev)
selector = RFE(LinearRegression(), n_features_to_select=15, step=5)

selector = selector.fit(X_train, y_train_log)
pd.DataFrame(dict(cols=X_train.columns, rank=selector.ranking_)).sort_values('rank').head(30)
len(X_train.columns[selector.ranking_ <= 2])
cols_selected = X_train.columns[selector.ranking_ <= 2]

cols_selected
reg = LinearRegression()

reg.fit(X_train[cols_selected], y_train_log)
y_pred = reg.predict(X_train[cols_selected])

mean_squared_error(y_train_log, y_pred)
y_pred_dev = reg.predict(X_dev[cols_selected])

mean_squared_error(y_dev_log, y_pred_dev)
sns.scatterplot(y_train_log, y_pred);
sns.scatterplot(y_dev_log, y_pred_dev);
from sklearn.ensemble import RandomForestRegressor
selector = RFE(RandomForestRegressor(), n_features_to_select=15, step=20)

selector = selector.fit(X_train, y_train)
cols_selected = X_train.columns[selector.ranking_ == 1]

cols_selected
reg = RandomForestRegressor(n_jobs=-1)

reg.fit(X_train[cols_selected], y_train)
y_pred = reg.predict(X_train[cols_selected])
sns.scatterplot(y_train, y_pred);
y_pred_dev = reg.predict(X_dev[cols_selected])
sns.scatterplot(y_dev, y_pred_dev);
from sklearn.metrics import r2_score
r2_score(y_train, y_pred)
r2_score(y_dev, y_pred_dev)
y_train.plot(kind='kde');
sns.distplot(np.log(1 + y_train));
reg = RandomForestRegressor()

reg.fit(X_train[cols_selected], y_train)