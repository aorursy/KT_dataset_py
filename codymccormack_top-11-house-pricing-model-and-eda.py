import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import math

import seaborn as sns

import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
def root_mean_squared_log_error(y_valid, y_preds):

    """Calculate root mean squared error of log(y_true) and log(y_pred)"""

    if len(y_preds)!=len(y_valid): return 'error_mismatch'

    y_preds_new = [math.log(x) for x in y_preds]

    y_valid_new = [math.log(x) for x in y_valid]

    return mean_squared_error(y_valid_new, y_preds_new, squared=False)
house_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

house_data.head()
house_data.shape
print(house_data.columns[house_data.isna().any()].tolist())

len(house_data.columns[house_data.isna().any()].tolist())
plt.rcParams['figure.figsize']=35,35

g = sns.heatmap(house_data.corr(),annot=True, fmt = ".1f", cmap = "coolwarm")
sns.barplot(x='YearRemodAdd', y='SalePrice', data=house_data)
sns.barplot(x='YearBuilt', y='SalePrice', data=house_data)
sns.barplot(x='LandSlope', y='SalePrice', data=house_data)
sns.barplot(x='LandContour', y='SalePrice', data=house_data)
sns.barplot(x='OverallQual', y='SalePrice', data=house_data)
sns.barplot(x='GarageCars', y='SalePrice', data=house_data)
sns.barplot(x='Fireplaces', y='SalePrice', data=house_data)
lot_price = house_data['LotArea'] + house_data['SalePrice']

sns.distplot(lot_price)
frontage_price = house_data['LotFrontage'] + house_data['SalePrice']

sns.distplot(frontage_price)
features = [x for x in house_data.columns if x not in ['SalePrice']]

X = house_data[features]

y = house_data['SalePrice']
'''X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)



numerical_cols = [cname for cname in X_train.columns if 

                X_train[cname].dtype in ['int64', 'float64']]



categorical_cols = [cname for cname in X_train.columns if

                    X_train[cname].nunique() < 13 and 

                    X_train[cname].dtype == "object"]





numerical_transformer = SimpleImputer(strategy='constant')



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

])'''
'''from sklearn.tree import DecisionTreeRegressor



tree_model = DecisionTreeRegressor(random_state=0)



tree_clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('tree_model', tree_model)

                     ])



tree_clf.fit(X_train, y_train)



tree_clf.fit(X_train, y_train)



tree_preds = tree_clf.predict(X_valid)



print('RMSLE:', root_mean_squared_log_error(y_valid, tree_preds))'''

#RMSLE: 0.20862182895771325
'''from sklearn.ensemble import RandomForestRegressor



random_model = RandomForestRegressor(random_state=0)



random_clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('random_model', random_model)

                     ])



random_clf.fit(X_train, y_train)



random_clf.fit(X_train, y_train)



random_preds = random_clf.predict(X_valid)



print('RMSLE:', root_mean_squared_log_error(y_valid, random_preds))'''

#RMSLE: 0.13715080920575703
'''xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.02, random_state=0)



xgb_clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('xgb_model', xgb_model)

                     ])



xgb_clf.fit(X_train, y_train, xgb_model__verbose=False)



xgb_clf.fit(X_train, y_train)



xgb_preds = xgb_clf.predict(X_valid)



print('RMSLE:', root_mean_squared_log_error(y_valid, xgb_preds))'''

#RMSLE: 0.13124779635294748
'''params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }



grid = GridSearchCV(xgb_model, param_grid=params, n_jobs=4, cv=5, verbose=3 )

grid.fit(param_X, y)

print('\n All results:')

print(grid.cv_results_)

print('\n Best estimator:')

print(grid.best_estimator_)

print('\n Best score:')

print(grid.best_score_ * 2 - 1)

print('\n Best parameters:')

print(grid.best_params_)'''
'''hp_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.6, gamma=0.5, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.02, max_delta_step=0, max_depth=4,

             min_child_weight=1, monotone_constraints='()',

             n_estimators=1000, n_jobs=0, num_parallel_tree=1, random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.8,

             tree_method='exact', validate_parameters=1, verbosity=None)



hp_clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('hp_model', hp_model)

                     ])



hp_clf.fit(X_train, y_train, hp_model__verbose=False)



hp_preds = hp_clf.predict(X_valid)



print('RMSLE:', root_mean_squared_log_error(y_valid, hp_preds))'''

#RMSLE: 0.12102431161864917

X.columns.to_list()
print(X['YearBuilt'].head())

print(X['YearRemodAdd'].head())
print(X['LotArea'].head())

print(X['LotFrontage'].head())
print(set(X['LandSlope']))

print(set(X['LandContour']))
print(set(X['YrSold']))

print(set(X['MoSold']))
print(set(X['Condition1']))

print(set(X['Condition2']))
print(set(X['ExterQual']))

print(set(X['ExterCond']))
print(set(X['YearBuilt']))

print(set(X['OverallQual']))
print(set(X['BedroomAbvGr']))

print(set(X['FullBath']))

print(set(X['HalfBath']))
X_feat_eng = X.copy()



X_feat_eng['years_since_update'] = X_feat_eng['YearRemodAdd'] - X_feat_eng['YearBuilt']

X_feat_eng['geometry'] = X_feat_eng['LotArea'] / X_feat_eng['LotFrontage']

X_feat_eng['land_topology'] = X_feat_eng['LandSlope'] + '_' + X_feat_eng['LandContour']

X_feat_eng['value_proposition'] = X_feat_eng['YearBuilt'] * X_feat_eng['OverallQual']

X_feat_eng['finished_basement'] = X_feat_eng['BsmtFinSF1'] > 0

X_feat_eng['garage_value'] = X_feat_eng['YearBuilt'] * X_feat_eng['GarageCars']

X_feat_eng['misc_value'] = X_feat_eng['Fireplaces'] + X_feat_eng['OverallQual']



X_feat_eng = X_feat_eng.drop(columns=['GarageCars'])



feature_numerical_cols = [cname for cname in X_feat_eng.columns if 

                X_feat_eng[cname].dtype in ['int64', 'float64']]



feature_categorical_cols = [cname for cname in X_feat_eng.columns if

                    X_feat_eng[cname].nunique() < 50 and 

                    X_feat_eng[cname].dtype in ['object', 'bool']]





feature_numerical_transformer = SimpleImputer(strategy='constant')



feature_categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



feature_preprocessor = ColumnTransformer(

    transformers=[

        ('num', feature_numerical_transformer, feature_numerical_cols),

        ('cat', feature_categorical_transformer, feature_categorical_cols)

])



feature_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.6, gamma=0.0, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.02, max_delta_step=0, max_depth=4,

             min_child_weight=0.0, monotone_constraints='()',

             n_estimators=1250, n_jobs=0, num_parallel_tree=1, random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.8,

             tree_method='exact', validate_parameters=1, verbosity=None)



feature_clf = Pipeline(steps=[('feature_preprocessor', feature_preprocessor),

                      ('feature_model', feature_model)

                     ])



feature_X_train, feature_X_valid, feature_y_train, feature_y_valid = train_test_split(X_feat_eng, y, random_state=0)



feature_clf.fit(feature_X_train, feature_y_train, feature_model__verbose=False) 

feature_preds = feature_clf.predict(feature_X_valid)



print('RMSLE:', root_mean_squared_log_error(feature_y_valid, feature_preds))



#RMSLE: 0.12003154815035846
X_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
X_test['years_since_update'] = X_test['YearRemodAdd'] - X_test['YearBuilt']

X_test['geometry'] = X_test['LotArea'] / X_test['LotFrontage']

X_test['land_topology'] = X_test['LandSlope'] + '_' + X_test['LandContour']

X_test['value_proposition'] = X_test['YearBuilt'] * X_test['OverallQual']

X_test['finished_basement'] = X_test['BsmtFinSF1'] > 0

X_test['garage_value'] = X_test['YearBuilt'] * X_test['GarageCars']

X_test['misc_value'] = X_test['Fireplaces'] + X_test['OverallQual']



X_test = X_test.drop(columns=['GarageCars'])



feature_clf.fit(X_feat_eng, y, feature_model__verbose=False)
preds = feature_clf.predict(X_test)

output = pd.DataFrame({'Id': X_test.Id,

                       'SalePrice': preds})

output.to_csv('submission.csv', index=False)