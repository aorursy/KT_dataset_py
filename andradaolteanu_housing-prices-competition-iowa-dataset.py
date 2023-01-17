#Imputs

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.pipeline import Pipeline
#Read the data

X = pd.read_csv('../input/iowa-house-prices/train.csv', index_col = 'Id')

X_test = pd.read_csv('../input/iowa-house-prices/test.csv', index_col ='Id')



#Filter from target column null values

X.dropna(axis = 0, subset = ['SalePrice'], inplace = True)

y = X['SalePrice']



#Filter out the target column from X dataset

X.drop(axis = 1, labels = ['SalePrice'], inplace = True)
#Verify if there are the same number of columns in both test and train data

print(X.shape)

print((X.columns == X_test.columns).sum())
X.columns
# MoSold, YrSOld, SaleType, SaleCondition won't be available for a prediction for the new house

#so these columns will be dropped

leakage_columns = ['MoSold', 'YrSold', 'SaleType', 'SaleCondition']

X.drop(labels = leakage_columns, axis = 1, inplace = True)

X_test.drop(labels = leakage_columns, axis = 1, inplace= True)
numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]



cols_with_nulls = X[numerical_cols].isnull().sum()

print(cols_with_nulls[cols_with_nulls > 0])
sns.set_style('whitegrid')

df = X[['LotFrontage', 'MasVnrArea', 'GarageYrBlt']].dropna(axis = 0)
#we'll impute the mode, as the distributions is asymetric and the mean is influenced by outliers

plt.figure(figsize = (12,4))

sns.distplot(a = df['LotFrontage'], bins = 30, norm_hist=False, kde=True, color = 'blue')
#we'll use most frequent, as the distribution is strongly asymetric to the right

plt.figure(figsize = (12,4))

sns.distplot(a = df['MasVnrArea'], bins = 30, norm_hist=False, kde=True, color = 'purple')
#for this distribution mean should work just fine :)

plt.figure(figsize = (12,4))

sns.distplot(a = df['GarageYrBlt'], bins = 30, norm_hist=False, kde=True, color = 'orange')
#There are only 3 columns with missing values, and they are small in number, so we'll apply Simple Imputation

from sklearn.impute import SimpleImputer



numerical_cols_median = ['LotFrontage']

numerical_transformer_median = SimpleImputer(strategy = 'median')



numerical_cols_mod = ['MasVnrArea']

numerical_transformer_mod = SimpleImputer(strategy = 'most_frequent')



numerical_cols_mean = ['GarageYrBlt']

numerical_transformer_mean = SimpleImputer(strategy = 'mean')



numerical_cols_remain = set(numerical_cols) - set(numerical_cols_mean) - set(numerical_cols_median) - set(numerical_cols_mod)

numerical_cols_remain = list(numerical_cols_remain)
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']

cols_with_nulls_categs = X[categorical_cols].isnull().sum()

cols_with_nulls_categs[cols_with_nulls_categs > 0]
#Because there are a few columns with too many missing values, we'll filter them out from the data

X.drop(labels = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)

X_test.drop(labels = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)
#Redo categorical_cols

categorical_cols = [col for col in X.columns if X[col].dtype == 'object']

X.shape
from sklearn.preprocessing import OneHotEncoder



categ_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy = 'most_frequent')),

                                         ('onehot', OneHotEncoder(handle_unknown = 'ignore'))])
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[('num_median', numerical_transformer_median, numerical_cols_median),

                                               ('num_mod', numerical_transformer_mod, numerical_cols_mod),

                                               ('num_mean', numerical_transformer_mean, numerical_cols_mean),

                                               ('num_rest', numerical_transformer_mean, numerical_cols_remain),

                                              ('cat', categ_transformer, categorical_cols)])
#split the training data into train & valid

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state = 0)
from xgboost import XGBRegressor

xgboost_1 = XGBRegressor(learning_rate = 0.05, n_estimators=1000, random_state=0)

pipeline_1 = Pipeline(steps=[('preprocessor', preprocessor),

                            ('model', xgboost_1)])

pipeline_1.fit(X_train, y_train)

preds_1 = pipeline_1.predict(X_valid)

mae = mean_absolute_error(y_valid, preds_1)

print('MAE 1:', mae)
X_train_prep = preprocessor.fit_transform(X_train)
from sklearn.model_selection import GridSearchCV



param_grid = {'learning_rate' : [0.1, 0.08, 0.05, 0.03, 0.01], 'n_estimators' : [50, 100, 200, 500, 700, 1000]}

grid = GridSearchCV(XGBRegressor(), param_grid, cv = 5, verbose=5)



grid.fit(X_train_prep, y_train)
print('Best params:', grid.best_params_)

print('Best estim:', grid.best_estimator_)

print('Best score:', grid.best_score_)
xgboost_2 = XGBRegressor(learning_rate=0.01, n_estimators= 1000)



pipeline_2 = Pipeline(steps=[('preprocessor', preprocessor),

                            ('model', xgboost_2)])



pipeline_2.fit(X_train, y_train)



preds_2 = pipeline_2.predict(X_valid)

mae = mean_absolute_error(y_valid, preds_2)

print('MAE 2:', mae)