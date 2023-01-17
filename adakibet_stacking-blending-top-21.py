import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action="ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
#drop id columns
train = train.drop('Id', axis = 1)
test = test.drop('Id', axis = 1)

#treatind year fields as objects to classify them as categorical data
train['YearBuilt'] =  train['YearBuilt'].astype(str)
train['YearRemodAdd'] =  train['YearRemodAdd'].astype(str)
train['YrSold'] =  train['YrSold'].astype(str)
test['YearBuilt'] =  test['YearBuilt'].astype(str)
test['YearRemodAdd'] =  test['YearRemodAdd'].astype(str)
test['YrSold'] =  test['YrSold'].astype(str)

#dependent and independent variables
train_independent = train.iloc[:, :-1]
train_dependent = train.iloc[:, -1]
#categorical and numerical features
numerical_features = []
categorical_features = []
for column in train_independent.columns:
    if train_independent[column].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
        numerical_features.append(column)
    elif train_independent[column].dtype == object:
        categorical_features.append(column)

columns = categorical_features + numerical_features
train_independent = train[columns]
test = test[columns]
#merging test and train independent variables
features = pd.concat([train_independent, test], axis = 0)
print('train features :', train_independent.shape)
print('train target :', train_dependent.shape)
print('test features :', test.shape)
print('train dataset :', train.shape)
print('all features :', features.shape)
#Numerical features with null values
null = features[numerical_features].isna().sum().sort_values(ascending = False)
null_values = pd.DataFrame(null)
null_values
features['GarageYrBlt'] = features['GarageYrBlt'].fillna(0)
features['MasVnrArea'] = features['MasVnrArea'].fillna(0)
features['BsmtFullBath'] = features['BsmtFullBath'].fillna(0)
features['BsmtHalfBath'] = features['BsmtHalfBath'].fillna(0)
features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)
features['BsmtUnfSF'] = features['BsmtUnfSF'].fillna(0)
features['BsmtFinSF2'] = features['BsmtFinSF2'].fillna(0)
features['GarageCars'] = features['GarageCars'].fillna(0)
features['GarageArea'] = features['GarageArea'].fillna(0)
features['BsmtFinSF1'] = features['BsmtFinSF1'].fillna(0)
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))

#checking for anymore missing values
features[numerical_features].isna().any()
#Categorical features with null values
null = features[categorical_features].isna().sum().sort_values(ascending = False)
null_values = pd.DataFrame(null)
null_values
features['PoolQC'] = features['PoolQC'].fillna('Null')
features['MiscFeature'] = features['MiscFeature'].fillna('Null')
features['Alley'] = features['Alley'].fillna('Null')
features['Fence'] = features['Fence'].fillna('Null')
features['FireplaceQu'] = features['FireplaceQu'].fillna('Null')
features['GarageCond'] = features['GarageCond'].fillna('Null')
features['GarageQual'] = features['GarageQual'].fillna('Null')
features['GarageFinish'] = features['GarageFinish'].fillna('Null')
features['GarageType'] = features['GarageType'].fillna('Null')
features['BsmtExposure'] = features['BsmtExposure'].fillna('Null')
features['BsmtCond'] = features['BsmtCond'].fillna('Null')
features['BsmtQual'] = features['BsmtQual'].fillna('Null')
features['BsmtFinType2'] = features['BsmtFinType2'].fillna('Null')
features['BsmtFinType1'] = features['BsmtFinType1'].fillna('Null')
features['Utilities'] = features['Utilities'].fillna('Null')
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])
features['Functional'] = features['Functional'].fillna(features['Functional'].mode()[0])
features['MasVnrType'] = features['MasVnrType'].fillna(features['MasVnrType'].mode()[0])
features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])

#checking for remaining null values
features[categorical_features].isna().any()
features['age'] = features['YrSold'].astype(int) - features['YearBuilt'].astype(int)
features['remod_age'] = features['YrSold'].astype(int) - features['YearRemodAdd'].astype(int)
features['extra_rooms'] = features['TotRmsAbvGrd'] - features['BedroomAbvGr'] - features['KitchenAbvGr']
features['floors_area'] = features['1stFlrSF'] + features['1stFlrSF']
features['total_bathrooms'] = features['FullBath'] + (0.5 * features['HalfBath']) + features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath'])
features['porch_area'] = features['WoodDeckSF'] + features['OpenPorchSF'] + features['EnclosedPorch'] + features['3SsnPorch'] + features['ScreenPorch'] + features['PoolArea']
features['walled_area'] = features['TotalBsmtSF'] +features['GrLivArea']
features['TotalOccupiedArea'] = features['walled_area'] + features['porch_area']
#dependent variable
f, ax = plt.subplots(figsize=(9, 8))
sns.distplot(train_dependent, bins = 20, color = 'Magenta')
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
# log transformation
train_dependent = np.log1p(train_dependent)

#after transformation
f, ax = plt.subplots(figsize=(9, 8))
sns.distplot(train_dependent, bins = 20, color = 'Magenta')
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
# Other numerical variables
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

skew_features = features[numerical_features].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features
# Normalize skewed features with boxcox transformation
for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
#defining numerical features again to include the added features for the correlation plot to be plotted.
numerical_features = []
for column in train_independent.columns:
    if train_independent[column].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
        numerical_features.append(column)

new_train_set = pd.concat([features.iloc[:len(train_dependent), :], train_dependent], axis=1)
def correlation_map(f_data, f_feature, f_number):
    f_most_correlated = f_data.corr().nlargest(f_number,f_feature)[f_feature].index
    f_correlation = f_data[f_most_correlated].corr()
    
    f_mask = np.zeros_like(f_correlation)
    f_mask[np.triu_indices_from(f_mask)] = True
    with sns.axes_style("white"):
        f_fig, f_ax = plt.subplots(figsize=(12, 10))
        f_ax = sns.heatmap(f_correlation, mask=f_mask, vmin=0, vmax=1, square=True,
                           annot=True, annot_kws={"size": 10}, cmap="BuPu")

    plt.show()

correlation_map(new_train_set, 'SalePrice', 20)
#dropping features with high correlation with other independent variables to avoid chances of multicolinearity.
features = features.drop(['GarageCars','1stFlrSF', 'walled_area'], axis = 1)
features = pd.get_dummies(features).reset_index(drop=True)
features.shape
features_to_be_dropped = []
for feature in features.columns:
    all_value_counts = features[feature].value_counts()
    zero_value_counts = all_value_counts.iloc[0]
    if zero_value_counts / len(features) > 0.995:
        features_to_be_dropped.append(feature)
print('\nFeatures with predominant zeroes:\n')
print(features_to_be_dropped)

features = features.drop(features_to_be_dropped, axis=1).copy()
features.shape
def logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   
        res.columns.values[m] = l + '_log'
        m += 1
    return res

log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                 'TotalBsmtSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                 'TotRmsAbvGrd','Fireplaces','GarageArea','WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch','3SsnPorch','ScreenPorch','MiscVal', 'floors_area']

features = logs(features, log_features)

def squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   
        res.columns.values[m] = l + '_sq'
        m += 1
    return res 

squared_features = ['OverallQual', 'LotFrontage_log', 
              'TotalBsmtSF_log', 'GrLivArea_log',
              'GarageArea_log', 'floors_area_log']
features = squares(features, squared_features)
x_train = features.iloc[:len(train_dependent), :]
x_test = features.iloc[len(train_dependent):, :]
y_train = train_dependent
train_set = pd.concat([x_train, y_train], axis=1)
print('train features:', x_train.shape)
print('train target:', y_train.shape)
print('test features:', x_test.shape)
print('train set:', train_set.shape)
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from mlxtend.regressor import StackingCVRegressor
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV


#randomforest
rf = RandomForestRegressor(n_estimators=300, random_state=0)

#adaboost
ada = AdaBoostRegressor(learning_rate = 0.05, loss =  'linear', n_estimators = 100 , random_state = 0)

#xgb
xgb = XGBRegressor(learning_rate=0.01, n_estimators=6000, max_depth=3, min_child_weight=0, gamma=0, subsample=0.7, colsample_bytree=0.7,
                       objective='reg:squarederror', nthread=-1, scale_pos_weight=1, seed=27, reg_alpha=0.00006, random_state=0)

#ridgecv
kfolds = KFold(n_splits=10, shuffle=True, random_state=0)
ridge = make_pipeline(RobustScaler(),
                      RidgeCV(alphas=[13.5, 14, 14.5, 15, 15.5], cv=kfolds))

#gradient
grad = GradientBoostingRegressor(n_estimators=4000, learning_rate=0.01, max_depth=4, max_features='sqrt', min_samples_leaf=15, 
                                 min_samples_split=10, loss='huber', random_state=0)

#stackcv 
stackcv = StackingCVRegressor(regressors=(rf, ada, xgb, 
                                          ridge, grad),
                              meta_regressor=xgb,
                              use_features_in_secondary=True)
#individual performance
scores_rf = -1 * cross_val_score(rf, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
scores_ada = -1 * cross_val_score(ada, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
scores_xgb = -1 * cross_val_score(xgb, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
scores_ridge = -1 * cross_val_score(ridge, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
scores_grad = -1 * cross_val_score(grad, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
print('random forest mae:', scores_rf.mean())
print('Ada boost:', scores_ada.mean())
print('xgboost:', scores_xgb.mean())
print('ridgecv:', scores_ridge.mean())
print('Gradient Boosting:', scores_grad.mean())
#fitting
rf_fit = rf.fit(x_train, y_train)
ada_fit = ada.fit(x_train, y_train)
xgb_fit = xgb.fit(x_train, y_train)
ridge_fit = ridge.fit(x_train, y_train)
grad_fit = grad.fit(x_train, y_train)

stackcv_fit = stackcv.fit(np.array(x_train), np.array(y_train))
blend = [0.1994, 0.0000, 0.2031, 0.2017, 0.2032, 0.2043]
#blending                
y_pred = np.expm1((blend[0] * rf_fit.predict(x_test)) +
                  (blend[1] * ada_fit.predict(x_test)) +
                  (blend[2] * xgb_fit.predict(x_test)) +
                  (blend[3] * ridge_fit.predict(x_test)) +
                  (blend[4] * grad_fit.predict(x_test)) +
                  (blend[5] * stackcv_fit.predict(np.array(x_test))))
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission.iloc[:, 1] = np.round_(y_pred)
submission.to_csv("submission_z.csv", index=False)