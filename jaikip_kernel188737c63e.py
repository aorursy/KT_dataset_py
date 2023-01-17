### Data Preprocessing/Exploration Imports

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import missingno as msno

%matplotlib inline



### Modeling Imports

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import scipy.stats as stats

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor

from xgboost import XGBRegressor

from sklearn import linear_model

from sklearn.linear_model import Lasso

from sklearn.metrics import r2_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error

from xgboost import XGBClassifier

from sklearn.linear_model import LinearRegression

import lightgbm as lgb

from lightgbm import LGBMRegressor

from mlxtend.regressor import StackingCVRegressor



from vecstack import stacking



### Ignore Warnings

import warnings

warnings.simplefilter('ignore')
### Reading in data 



train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test.describe()
train.describe()
train.shape
test.shape
train_numeric_features = train.select_dtypes(include = [np.number])

test_numeric_features = test.select_dtypes(include = [np.number])
train_numeric_features.columns.size
test_numeric_features.columns.size
train_categorical_features = train.select_dtypes(include = [np.object])

test_categorical_features = test.select_dtypes(include = [np.object])
train_categorical_features.columns.size
test_categorical_features.columns.size
f, ax = plt.subplots(figsize=(11, 5))

saleprice_plot = sns.distplot(train['SalePrice'], color = 'red')

saleprice_plot
train['SalePrice'].skew()
train['SalePrice'].kurt()
### Using GrLivArea was recommended by the host of this competition

train = train[train.GrLivArea < 4500]

train.reset_index(drop = True, inplace = True)

train["SalePrice"] = np.log1p(train["SalePrice"])
f, ax = plt.subplots(figsize=(11, 5))

saleprice_norm = sns.distplot(train['SalePrice'], color = 'red')

saleprice_norm
train['SalePrice'].skew()
train['SalePrice'].kurt()
f, ax = plt.subplots(figsize=(15,12))

sns.heatmap(train.corr().abs(), square = True,

            fmt = '.2f', vmax = 0.8, 

            cmap = 'Reds', annot = True,

            annot_kws = {'size': 6})



### BELOW CODE IS COPIED FROM A FORUM TO FIX A MATPLOTLIB/SEABORN ISSUE###

### FORUM: https://github.com/mwaskom/seaborn/issues/1773 ###

b, t = plt.ylim()

b += 0.5 

t -= 0.5 

plt.ylim(b, t) 

plt.show()
f, ax = plt.subplots(figsize=(15,12))

k = 10

cols = train.corr().abs().nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale = 1.25)

sns.heatmap(cm, cbar = True, 

            annot = True, square = True, 

            fmt = '.2f', cmap = 'Reds', 

            annot_kws = {'size': 12}, 

            yticklabels = cols.values, 

            xticklabels = cols.values)



### BELOW CODE IS COPIED FROM A FORUM TO FIX A MATPLOTLIB/SEABORN ISSUE###

### FORUM: https://github.com/mwaskom/seaborn/issues/1773 ###

b, t = plt.ylim()

b += 0.5 

t -= 0.5 

plt.ylim(b, t) 

plt.show()
msno.matrix(train)
msno.matrix(test)
def fill_missing_values(data):

    missing_data = data.isnull().sum()

    missing_data = missing_data[missing_data > 0]

    

    for col in list(missing_data.index):

        if data[col].dtype == 'object':

            data[col].fillna(data[col].value_counts().index[0], inplace=True)

        elif data[col].dtype == 'int' or 'float':

            data[col].fillna(data[col].median(), inplace=True)
fill_missing_values(train)

fill_missing_values(test)
train_imputed = train.copy()

test_imputed = test.copy()
for col in train_numeric_features.columns:

    train_imputed[col] = train_imputed[col][(np.abs(stats.zscore(train_imputed[col])) < 3)]

    

for col in test_numeric_features.columns:

    test_imputed[col] = test_imputed[col][(np.abs(stats.zscore(test_imputed[col])) < 3)]
fill_missing_values(train_imputed)

fill_missing_values(test_imputed)
train_dummification = train_imputed.copy()

test_dummification = test_imputed.copy()
def convert_non_numerics(df):

    object_cols = list(df.select_dtypes(exclude=[np.number]).columns)

    object_cols_ind = []

    for col in object_cols:

        object_cols_ind.append(df.columns.get_loc(col))

    

    label_enc = LabelEncoder()

    for i in object_cols_ind:

        df.iloc[:,i] = label_enc.fit_transform(df.iloc[:,i])
convert_non_numerics(train_dummification)

convert_non_numerics(test_dummification)
X = train_dummification.drop('SalePrice', axis=1)

y = np.ravel(np.array(train_dummification[['SalePrice']].reset_index(drop = True)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
rf = RandomForestRegressor(n_estimators = 1200,

                          max_depth = 15,

                          min_samples_split = 5,

                          min_samples_leaf = 5,

                          max_features = None,

                          oob_score = True,

                          random_state = 50).fit(X_train, y_train)
gbr = GradientBoostingRegressor(n_estimators = 6250,

                                learning_rate = 0.01,

                                max_depth = 4,

                                max_features = 'sqrt',

                                min_samples_leaf = 15,

                                min_samples_split = 10,

                                loss = 'huber',

                                random_state = 50).fit(X_train, y_train)
xgboost = XGBRegressor(learning_rate = 0.01,

                       n_estimators = 6250,

                       max_depth = 4,

                       min_child_weight = 0,

                       gamma = 0.6,

                       subsample = 0.7,

                       colsample_bytree = 0.7,

                       objective = 'reg:linear',

                       nthread = -1,

                       scale_pos_weight = 1,

                       seed = 27,

                       reg_alpha = 0.00006,

                       random_state = 50).fit(X_train, y_train)
lasso = Lasso(alpha = 0.0025, fit_intercept = True, 

              normalize = False, precompute = False, 

              copy_X = True, max_iter = 1000, 

              tol = 0.0001, warm_start = False, 

              positive = False, random_state = 50).fit(X_train, y_train)
scores = cross_val_score(rf, X, y, cv = 5)

print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(gbr, X, y, cv = 5)

print("Gradient Boosting Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(xgboost, X, y, cv = 5)

print("XGBoost Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(lasso, X, y, cv = 5)

print("Lasso Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
final_labels = (np.exp(rf.predict(X_test)) + 

                np.exp(gbr.predict(X_test)) +

                np.exp(xgboost.predict(X_test)) + 

                np.exp(lasso.predict(X_test))) / 4
X_test['SalePrice'] = final_labels
X_test['SalePrice'].describe()
f, ax = plt.subplots(figsize=(15,12))

k = 10

cols = X_test.corr().abs().nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(X_test[cols].values.T)

sns.set(font_scale = 1.25)

sns.heatmap(cm, cbar = True, 

            annot = True, square = True, 

            fmt = '.2f', cmap = 'Reds', 

            annot_kws = {'size': 12}, 

            yticklabels = cols.values, 

            xticklabels = cols.values)



### BELOW CODE IS COPIED FROM A FORUM TO FIX A MATPLOTLIB/SEABORN ISSUE###

### FORUM: https://github.com/mwaskom/seaborn/issues/1773 ###

b, t = plt.ylim()

b += 0.5 

t -= 0.5 

plt.ylim(b, t) 

plt.show()
# https://www.kaggle.com/jaikip/kernel188737c63e/edit/run/24087615