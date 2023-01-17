# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



pd.set_option('display.max_columns', None)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATA_DIR = os.path.abspath('../input/home-data-for-ml-course/')

TRAIN_DIR = os.path.join(DATA_DIR, 'train.csv')

TEST_DIR = os.path.join(DATA_DIR, 'test.csv')



housing = pd.read_csv(TRAIN_DIR, index_col = 'Id')



housing = housing.dropna(subset = ['SalePrice'])

housing['SalePrice'] = np.log1p(housing['SalePrice'])
housing.head()
missing = housing.isnull().mean()*100

missing = missing[missing>0].sort_values(ascending = False)

missing.plot.bar()
# drop columns where more than 40% of data is unavailable

cols_to_drop = [column for column in list(housing) if housing[column].isnull().mean() > 0.4]

housing = housing.drop(cols_to_drop, axis = 1)
from sklearn.model_selection import train_test_split



housing_train, housing_valid = train_test_split(housing, random_state = 42)



# list of categorical feature names

attr_cat = [column for column in list(housing_train) if (housing_train[column].dtype == 'object' and column != 'SalePrice')]



# list of numerical feature names

attr_num = [column for column in list(housing_train) if (housing_train[column].dtype in ['int64', 'float64'] and column != 'SalePrice')]
import matplotlib.pyplot as plt

import seaborn as sns



fig = plt.figure(figsize=(18,24))

for i, attr in enumerate(attr_cat):

    fig.add_subplot(8,5,i+1)

    sns.countplot(housing_train[attr])

    plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
cat_to_drop = ['Street', 'LandContour', 'Utilities', 'LandSlope', 'Condition1', 'Condition2', 'RoofMatl', 'Heating', 'SaleType', 'PavedDrive', 'GarageCond', 'GarageQual', 'Functional', 'Electrical', 'CentralAir', 'Heating', 'BsmtFinType2', 'BsmtCond','ExterCond']



housing_train = housing_train.drop(cat_to_drop, axis = 1)

attr_cat = list(set(attr_cat) - set(cat_to_drop))
fig = plt.figure(figsize=(18,24))

for i, attr in enumerate(attr_num):

    fig.add_subplot(10,6,i+1)

    sns.distplot(housing_train[attr], kde=False)

plt.tight_layout()

plt.show()
num_to_drop = ['3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'EnclosedPorch', 'KitchenAbvGr', 'LowQualFinSF', 'BsmtHalfBath', 'BsmtFinSF2']



housing_train = housing_train.drop(num_to_drop, axis = 1)

attr_num = list(set(attr_num) - set(num_to_drop))
fig = plt.figure(figsize=(24,28))

for i, attr in enumerate(attr_num):

    fig.add_subplot(10,6,i+1)

    sns.boxplot(y = housing_train[attr])

plt.tight_layout()

plt.show()
attr_num_out = ['LotArea', 'OpenPorchSF', 'WoodDeckSF', 'BsmtUnfSF', 'BsmtFinSF1', 'LotFrontage', 'GarageArea', 'TotalBsmtSF', 'GrLivArea','MasVnrArea']

housing_train[attr_num_out].describe()
housing_train = housing_train.drop(housing_train[housing_train['LotArea']>25000].index)

housing_train = housing_train.drop(housing_train[housing_train['OpenPorchSF']>150].index)

housing_train = housing_train.drop(housing_train[housing_train['WoodDeckSF']>400].index)

housing_train = housing_train.drop(housing_train[housing_train['BsmtUnfSF']>1600].index)

housing_train = housing_train.drop(housing_train[housing_train['BsmtFinSF1']>1400].index)

housing_train = housing_train.drop(housing_train[housing_train['LotFrontage']>160].index)

housing_train = housing_train.drop(housing_train[housing_train['GarageArea']>1200].index)

housing_train = housing_train.drop(housing_train[housing_train['TotalBsmtSF']>2600].index)

housing_train = housing_train.drop(housing_train[housing_train['GrLivArea']>3600].index)

housing_train = housing_train.drop(housing_train[housing_train['MasVnrArea']>320].index)
num_correlation = housing_train[attr_num].corr()

plt.figure(figsize=(20,20))

plt.title('High Correlation')

sns.heatmap(num_correlation > 0.8, annot=True, square=True)
attr_num_corr = ['GarageCars', 'TotRmsAbvGrd', 'GarageYrBlt']

housing_train = housing_train.drop(attr_num_corr, axis = 1)

attr_num = list(set(attr_num) - set(attr_num_corr))
plt.figure(figsize=(20,20))



fig = plt.figure(figsize=(20,20))

for i, attr in enumerate(attr_num):

    fig.add_subplot(10,6,i+1)

    sns.scatterplot(x = attr, y = 'SalePrice', data = housing_train)

plt.tight_layout()

plt.show()

## 1 - categorical pipeline



from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



cat_pipeline = Pipeline(steps = [

    ('impute', SimpleImputer(strategy = 'most_frequent')),

    ('encode', OneHotEncoder(handle_unknown='ignore'))

])





## 2 - numerical pipeline



from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

    ('impute', SimpleImputer(strategy = 'mean')),

    ('scale', StandardScaler()),

])





## whole preprocessing pipeline



from sklearn.compose import ColumnTransformer



preprocessing_pipeline = ColumnTransformer([

    ('cat', cat_pipeline, attr_cat),

    ('num', num_pipeline, attr_num)

])



# having feature importances by fitting a basic XGBRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV

import scipy.stats as stats



xgb_reg = XGBRegressor(n_jobs = -1, random_state = 123)





# X_train, y_train, X_valid, y_valid



attr = attr_cat + attr_num



X_train = housing_train[attr]

y_train = housing_train['SalePrice']

X_valid = housing_valid[attr]

y_valid = housing_valid['SalePrice']



X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)

X_valid_preprocessed = preprocessing_pipeline.transform(X_valid)





xgb_reg.fit(X_train_preprocessed, y_train)



feature_importances = xgb_reg.feature_importances_



from sklearn.base import BaseEstimator, TransformerMixin



def indices_of_top_k(arr, k):

    return np.sort(np.argpartition(np.array(arr), -k)[-k:])



class TopFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_importances, k):

        self.feature_importances = feature_importances

        self.k = k

    def fit(self, X, y=None):

        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)

        return self

    def transform(self, X):

        return X[:, self.feature_indices_]
# Creating whole pipeline by adding feature selection to the pipeline



estimator = Pipeline([

    ('preprocessing', preprocessing_pipeline),

    ('feature_selection', TopFeatureSelector(feature_importances, k=20)),

    ('inference', xgb_reg)

])



# preprocessing parameter grid

param_dist = {'strategy': ['most_frequent', 'median', 'mean']

             }



preprocessing_param_grid = {'preprocessing__num__impute__'+k:v for k,v in param_dist.items()}





# feature selection parameter grid

param_dist = {'k': stats.randint(10, 30),

             }



feature_selection_param_grid = {'feature_selection__'+k:v for k,v in param_dist.items()}





# inference parameter grid

param_dist = {'n_estimators': stats.randint(1000, 4000),

              'learning_rate': stats.expon(0.005, 0.1),

             }



inference_param_grid = {'inference__'+k:v for k,v in param_dist.items()}



param_grid = {**preprocessing_param_grid, **feature_selection_param_grid, **inference_param_grid}



estimator_opt = RandomizedSearchCV(estimator, param_grid, n_jobs = -1,n_iter = 50, scoring = 'neg_mean_absolute_error', cv = 5, random_state = 123, return_train_score = True)





estimator_opt.fit(X_train, y_train)

preds = estimator_opt.predict(X_valid)



from sklearn.metrics import mean_squared_error



## Inference on test set



X_test = pd.read_csv(TEST_DIR, index_col = 'Id')



test_preds = estimator_opt.predict(X_test)



submission = pd.DataFrame({'Id': X_test.index , 'SalePrice' : np.exp(test_preds)})



submission.to_csv('submission.csv', index = False)