# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
house_price_train_dataset = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
house_price_test_dataset = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
pd.pandas.set_option('display.max_columns', None)
house_price_train_dataset.head()
pd.pandas.set_option('display.max_columns', None)
house_price_test_dataset.head()
house_price_train_dataset.shape
house_price_test_dataset.shape
house_price_train_dataset.info()
house_price_test_dataset.info()
train_features_with_nan = [feature for feature in house_price_train_dataset.columns if (house_price_train_dataset[feature].isnull().sum() > 0)]
plt.figure(figsize=(20,5))
house_price_train_dataset[train_features_with_nan].isnull().sum().plot(kind='bar')
test_features_with_nan = [feature for feature in house_price_test_dataset.columns if (house_price_test_dataset[feature].isnull().sum() > 0)]
plt.figure(figsize=(20,5))
house_price_test_dataset[test_features_with_nan].isnull().sum().plot(kind='bar')
plt.figure(figsize=(20,5))
house_price_train_dataset[train_features_with_nan].isnull().mean().plot(kind='bar')
plt.figure(figsize=(20,5))
house_price_test_dataset[test_features_with_nan].isnull().mean().plot(kind='bar')
plt.figure(figsize=(30,10))
sns.heatmap(house_price_train_dataset.isnull(), cbar=False)
plt.show()
plt.figure(figsize=(30,10))
sns.heatmap(house_price_test_dataset.isnull(), cbar=False)
plt.show()
msno.heatmap(house_price_train_dataset)
msno.heatmap(house_price_test_dataset)
Id = house_price_test_dataset['Id']
drop_columns = ['Id', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
house_price_train_dataset.drop(house_price_train_dataset[drop_columns], inplace=True, axis=1)
house_price_test_dataset.drop(house_price_test_dataset[drop_columns], inplace=True, axis=1)
house_price_train_dataset.shape,house_price_test_dataset.shape
train_cat_features_with_nan = [feature for feature in house_price_train_dataset if (house_price_train_dataset[feature].isnull().sum() > 0 and
                                                                                house_price_train_dataset[feature].dtypes == 'O')]
plt.figure(figsize=(20,5))
house_price_train_dataset[train_cat_features_with_nan].isnull().sum().plot(kind='bar')
plt.show()
test_cat_features_with_nan = [feature for feature in house_price_test_dataset if (house_price_test_dataset[feature].isnull().sum() > 0 and
                                                                                house_price_test_dataset[feature].dtypes == 'O')]
plt.figure(figsize=(20,5))
house_price_test_dataset[test_cat_features_with_nan].isnull().sum().plot(kind='bar')
plt.show()
train_num_features_with_nan = [feature for feature in house_price_train_dataset.columns if (house_price_train_dataset[feature].isnull().sum() > 0 and
                                                                                house_price_train_dataset[feature].dtypes != 'O')]
house_price_train_dataset[train_num_features_with_nan].isnull().sum().plot(kind='bar')
test_num_features_with_nan = [feature for feature in house_price_test_dataset.columns if (house_price_test_dataset[feature].isnull().sum() > 0 and
                                                                                house_price_test_dataset[feature].dtypes != 'O')]
house_price_test_dataset[test_num_features_with_nan].isnull().sum().plot(kind='bar')
train_num_features_with_outliers = [feature for feature in house_price_train_dataset.columns if (house_price_train_dataset[feature].dtypes != 'O')]
house_price_train_dataset[train_num_features_with_outliers].describe()
test_num_features_with_outliers = [feature for feature in house_price_test_dataset.columns if (house_price_test_dataset[feature].dtypes != 'O')]
house_price_test_dataset[test_num_features_with_outliers].describe()
house_price_train_dataset.boxplot(column='MSSubClass')
house_price_train_dataset.boxplot(column='LotFrontage')
house_price_train_dataset.boxplot(column='LotArea')
house_price_train_dataset.boxplot(column='MasVnrArea')
house_price_train_dataset.boxplot(column='BsmtFinSF1')
outliers_features = ['MSSubClass', 'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch']
outliers_features
plt.figure(figsize=(60,20))
plt.plot(house_price_train_dataset.skew())
plt.figure(figsize=(60,20))
plt.plot(house_price_test_dataset.skew())
house_price_train_dataset.duplicated().sum()
house_price_test_dataset.duplicated().sum()
def replace_cat_missing_values_in_train_dataset(house_price_train_dataset, train_cat_features_with_nan):
    house_price_train_dataset_new = house_price_train_dataset.copy()
    for feature in train_cat_features_with_nan:
        house_price_train_dataset_new[feature] = house_price_train_dataset_new[feature].fillna(house_price_train_dataset_new[feature].mode()[0])
    return house_price_train_dataset_new

house_price_train_dataset = replace_cat_missing_values_in_train_dataset(house_price_train_dataset, train_cat_features_with_nan)

def replace_cat_missing_values_in_test_dataset(house_price_test_dataset, test_cat_features_with_nan):
    house_price_test_dataset_new = house_price_test_dataset.copy()
    for feature in test_cat_features_with_nan:
        house_price_test_dataset_new[feature] = house_price_test_dataset_new[feature].fillna(house_price_test_dataset_new[feature].mode()[0])
    return house_price_test_dataset_new

house_price_test_dataset = replace_cat_missing_values_in_test_dataset(house_price_test_dataset, test_cat_features_with_nan)
for feature in train_num_features_with_nan:
    median_of_feature = house_price_train_dataset[feature].median()
    house_price_train_dataset[feature].fillna(median_of_feature, inplace=True)
    
for feature in test_num_features_with_nan:
    median_of_feature = house_price_test_dataset[feature].median()
    house_price_test_dataset[feature].fillna(median_of_feature, inplace=True)
plt.figure(figsize=(30,5))
plt.plot(house_price_train_dataset.isnull().sum())

plt.figure(figsize=(30,5))
plt.plot(house_price_test_dataset.isnull().sum())
for feature in outliers_features:
    IQR = house_price_train_dataset[feature].quantile(0.75) - house_price_train_dataset[feature].quantile(0.25)
    lower_boundary = house_price_train_dataset[feature].quantile(0.25) - (IQR*3)
    upper_boundary = house_price_train_dataset[feature].quantile(0.75) + (IQR*3)
    print(feature, lower_boundary, upper_boundary)
for feature in outliers_features:
    IQR = house_price_train_dataset[feature].quantile(0.75) - house_price_train_dataset[feature].quantile(0.25)
    lower_boundary = house_price_train_dataset[feature].quantile(0.25) - (IQR*3)
    upper_boundary = house_price_train_dataset[feature].quantile(0.75) + (IQR*3)
    house_price_train_dataset.loc[house_price_train_dataset[feature]<=lower_boundary, feature] = lower_boundary
    house_price_train_dataset.loc[house_price_train_dataset[feature]>=upper_boundary, feature] = upper_boundary
    
house_price_train_dataset[outliers_features].describe()
for feature in outliers_features:
    IQR = house_price_test_dataset[feature].quantile(0.75) - house_price_test_dataset[feature].quantile(0.25)
    lower_boundary = house_price_test_dataset[feature].quantile(0.25) - (IQR*3)
    upper_boundary = house_price_test_dataset[feature].quantile(0.75) + (IQR*3)
    print(feature, lower_boundary, upper_boundary)
for feature in outliers_features:
    IQR = house_price_test_dataset[feature].quantile(0.75) - house_price_test_dataset[feature].quantile(0.25)
    lower_boundary = house_price_test_dataset[feature].quantile(0.25) - (IQR*3)
    upper_boundary = house_price_test_dataset[feature].quantile(0.75) + (IQR*3)
    house_price_test_dataset.loc[house_price_test_dataset[feature]<=lower_boundary, feature] = lower_boundary
    house_price_test_dataset.loc[house_price_test_dataset[feature]>=upper_boundary, feature] = upper_boundary
    
house_price_test_dataset[outliers_features].describe()
plt.figure(figsize=(30,20))
corr = house_price_train_dataset.corr()
sns.heatmap(corr, annot=True, cmap='RdYlGn')
plt.show()
corr_features = corr.index[abs(corr['SalePrice']) < 0.3]
corr_features
house_price_train_dataset.drop(['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', 'HalfBath', 'KitchenAbvGr', 'KitchenAbvGr', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'], axis=1, inplace=True)
house_price_test_dataset.drop(['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', 'HalfBath', 'KitchenAbvGr', 'KitchenAbvGr', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'], axis=1, inplace=True)
house_price_train_dataset.shape, house_price_test_dataset.shape
house_price_train_dataset['train'] = 1
house_price_test_dataset['train'] = 0
house_price_dataset = pd.concat([house_price_train_dataset, house_price_test_dataset], axis=0)
house_price_dataset.shape
house_price_dataset = pd.get_dummies(house_price_dataset, drop_first=True)
house_price_dataset.shape
train_dataset = house_price_dataset[house_price_dataset['train'] == 1]
test_dataset = house_price_dataset[house_price_dataset['train'] == 0]
train_dataset.shape, test_dataset.shape
train_dataset.drop(['train'], axis=1, inplace=True)
test_dataset.drop(['train'], axis=1, inplace=True)
train_dataset.shape, test_dataset.shape
test_dataset.drop(['SalePrice'], axis=1, inplace=True)
train_features = train_dataset.drop(['SalePrice'], axis=1)
train_label = train_dataset['SalePrice']
train_features.shape
plt.figure(figsize=(12,10))
from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor()
etr.fit(train_features, train_label)
feat_importances = pd.Series(etr.feature_importances_, index=train_features.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=20, random_state=None, shuffle=False)
skf.get_n_splits(train_features, train_label)
for train_index, test_index in skf.split(train_features, train_label):
    features_train, features_test = train_features.iloc[train_index], train_features.iloc[test_index]
    label_train, label_test = train_label.iloc[train_index], train_label.iloc[test_index]
#Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]

#Number of features to consider in every split
max_features = ['auto', 'sqrt']

#Maximum number of levels in a tree
max_depth = [int(x) for x in np.linspace(start=5, stop=30, num=6)]

#Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]

#Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
#Random Grid
random_grid = {'n_estimators' : n_estimators,
              'max_features' : max_features,
              'max_depth' : max_depth,
              'min_samples_split' : min_samples_split,
              'min_samples_leaf' : min_samples_leaf}
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
random_forest = RandomForestRegressor()
randam_forest_model = RandomizedSearchCV(estimator=random_forest, param_distributions=random_grid,
                                         scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2,
                                        random_state=42, n_jobs=1)
randam_forest_model.fit(train_features, train_label)
randam_forest_model.best_params_
from sklearn.metrics import r2_score
r2_score(label_train, randam_forest_model.predict(features_train))
from sklearn.metrics import r2_score
r2_score(label_test, randam_forest_model.predict(features_test))
plt.figure(figsize=(12,5))
sns.distplot(label_train-randam_forest_model.predict(features_train))
plt.figure(figsize=(12,5))
sns.distplot(label_test-randam_forest_model.predict(features_test))
test_label = randam_forest_model.predict(test_dataset)
test_label
house_price_predictions = pd.DataFrame(test_label)
house_price_prediction_submission = pd.concat([Id, house_price_predictions], axis=1)
house_price_prediction_submission.columns = ['Id', 'SalePrice']
house_price_prediction_submission.to_csv('house_price_prediction_submission.csv', index=False)