import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor

sns.set()
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
start_time = datetime.now()
def assessment(f_data, f_y_feature, f_x_feature, f_index=-1):
    """
    Develops and displays a histogram and a scatter plot for a dependent / independent variable pair from
    a dataframe and, optionally, highlights a specific observation on the plot in a different color (red).
    
    Also optionally, if an independent feature is not informed, the scatterplot is not displayed.
    
    Keyword arguments:
    
    f_data      Tensor containing the dependent / independent variable pair.
                Pandas dataframe
    f_y_feature Dependent variable designation.
                String
    f_x_feature Independent variable designation.
                String
    f_index     If greater or equal to zero, the observation denoted by f_index will be plotted in red.
                Integer
    """
    for f_row in f_data:
        if f_index >= 0:
            f_color = np.where(f_data[f_row].index == f_index,'r','g')
            f_hue = None
        else:
            f_color = 'b'
            f_hue = None
    
    f_fig, f_a = plt.subplots(1, 2, figsize=(16,4))
    
    f_chart1 = sns.distplot(f_data[f_x_feature], ax=f_a[0], kde=False, color='r')
    f_chart1.set_xlabel(f_x_feature,fontsize=10)
    
    if f_index >= 0:
        f_chart2 = plt.scatter(f_data[f_x_feature], f_data[f_y_feature], c=f_color, edgecolors='w')
        f_chart2 = plt.xlabel(f_x_feature, fontsize=10)
        f_chart2 = plt.ylabel(f_y_feature, fontsize=10)
    else:
        f_chart2 = sns.scatterplot(x=f_x_feature, y=f_y_feature, data=f_data, hue=f_hue, legend=False)
        f_chart2.set_xlabel(f_x_feature,fontsize=10)
        f_chart2.set_ylabel(f_y_feature,fontsize=10)

    plt.show()


def correlation_map(f_data, f_feature, f_number):
    """
    Develops and displays a heatmap plot referenced to a primary feature of a dataframe, highlighting
    the correlation among the 'n' mostly correlated features of the dataframe.
    
    Keyword arguments:
    
    f_data      Tensor containing all relevant features, including the primary.
                Pandas dataframe
    f_feature   The primary feature.
                String
    f_number    The number of features most correlated to the primary feature.
                Integer
    """
    f_most_correlated = f_data.corr().nlargest(f_number,f_feature)[f_feature].index
    f_correlation = f_data[f_most_correlated].corr()
    
    f_mask = np.zeros_like(f_correlation)
    f_mask[np.triu_indices_from(f_mask)] = True
    with sns.axes_style("white"):
        f_fig, f_ax = plt.subplots(figsize=(12, 10))
        f_ax = sns.heatmap(f_correlation, mask=f_mask, vmin=0, vmax=1, square=True,
                           annot=True, annot_kws={"size": 10}, cmap="BuPu")

    plt.show()
    
    
def xval_rmse_scoring(f_model, f_X, f_y, f_cv):
    """
    Returns a machine learning model cross-validated score based on the Root Mean Squared Error (RMSE) metric.
    
    Keyword arguments:
    
    f_model     Machine learning model.
                Object instance
    f_X_        Tensor containing features for modeling.
                Pandas dataframe
    f_y         Tensor containing targets for modeling.
                Pandas series
    f_cv        Cross-validation splitting strategy.
                Please refer to scikit-learn's model_selection cross_val_score for further information.
    """
    return np.sqrt(-cross_val_score(f_model, f_X, f_y,
                                    scoring='neg_mean_squared_error',
                                    cv=f_cv))

train = pd.read_csv('../input/home-data-for-ml-course/train.csv')
test = pd.read_csv('../input/home-data-for-ml-course/test.csv')

train.rename(columns={"1stFlrSF": "FstFlSF", "2ndFlrSF": "SecFlSF", "3SsnPorch": "ThreeSPorch"}, inplace=True)
test.rename(columns={"1stFlrSF": "FstFlSF", "2ndFlrSF": "SecFlSF", "3SsnPorch": "ThreeSPorch"}, inplace=True)

train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

train_features = train.iloc[:, :-1]
train_targets = train.iloc[:, -1]
test_features = test
test_features.iloc[1132,58] = 2007
train_features['MSSubClass'] = train_features['MSSubClass'].astype(str)
train_features['MoSold'] = train_features['MoSold'].astype(str)
train_features['YrSold'] = train_features['YrSold'].astype(str)
test_features['MSSubClass'] = test_features['MSSubClass'].astype(str)
test_features['MoSold'] = test_features['MoSold'].astype(str)
test_features['YrSold'] = test_features['YrSold'].astype(str)
features_numerical = []
features_categorical = []
for column in train_features.columns:
    if train_features[column].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
        features_numerical.append(column)
    elif train_features[column].dtype == object:
        features_categorical.append(column)

new_order = features_numerical + features_categorical
train_features = train_features[new_order]
test_features = test_features[new_order]
train_test_features = pd.concat([train_features, test_features], axis=0)
missing_numerical = []
for feature in features_numerical:
    if train_test_features.count()[feature] < train_test_features.shape[0]:
        missing_numerical.append(feature)
print(f'\nNumerical features with missing values before treatment: {len(missing_numerical)}\n')
print(missing_numerical)
train_test_features['LotFrontage'] = train_test_features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
train_test_features['MasVnrArea'] = train_test_features['MasVnrArea'].fillna(0)
train_test_features['BsmtFinSF1'] = train_test_features['BsmtFinSF1'].fillna(0)
train_test_features['BsmtFinSF2'] = train_test_features['BsmtFinSF2'].fillna(0)
train_test_features['BsmtUnfSF'] = train_test_features['BsmtUnfSF'].fillna(0)
train_test_features['TotalBsmtSF'] = train_test_features['TotalBsmtSF'].fillna(0)
train_test_features['BsmtFullBath'] = train_test_features['BsmtFullBath'].fillna(0)
train_test_features['BsmtHalfBath'] = train_test_features['BsmtHalfBath'].fillna(0)
train_test_features['GarageYrBlt'] = train_test_features['GarageYrBlt'].fillna(0)
train_test_features['GarageArea'] = train_test_features['GarageArea'].fillna(0)
train_test_features['GarageCars'] = train_test_features['GarageCars'].fillna(0)

missing_numerical.clear()
for feature in features_numerical:
    if train_test_features.count()[feature] < train_test_features.shape[0]:
        missing_numerical.append(feature)
print(f'\nNumerical features with missing values after treatment: {len(missing_numerical)}')
train_test_features['YearsSinceBuilt'] = train_test_features['YrSold'].astype(int) - train_test_features['YearBuilt']
train_test_features['YearsSinceRemod'] = train_test_features['YrSold'].astype(int) - train_test_features['YearRemodAdd']

train_test_features['TotalWalledArea'] = train_test_features['TotalBsmtSF'] + train_test_features['GrLivArea']
train_test_features['TotalPorchArea'] = train_test_features['OpenPorchSF'] + train_test_features['ThreeSPorch'] + train_test_features['EnclosedPorch'] + train_test_features['ScreenPorch'] + train_test_features['WoodDeckSF']
train_test_features['TotalOccupiedArea'] = train_test_features['TotalWalledArea'] + train_test_features['TotalPorchArea']

train_test_features['OtherRooms'] = train_test_features['TotRmsAbvGrd'] - train_test_features['BedroomAbvGr'] - train_test_features['KitchenAbvGr']
train_test_features['TotalBathrooms'] = train_test_features['FullBath'] + (0.5 * train_test_features['HalfBath']) + train_test_features['BsmtFullBath'] + (0.5 * train_test_features['BsmtHalfBath'])

train_test_features['LotDepth'] = train_test_features['LotArea'] / train_test_features['LotFrontage']
skew_values = train_test_features[features_numerical].apply(lambda x: skew(x))
high_skew = skew_values[skew_values > 0.5]
skew_indices = high_skew.index
for index in skew_indices:
    assessment(pd.concat([train_test_features.iloc[:len(train_targets), :], train_targets], axis=1), 'SalePrice', index, -1)
for index in skew_indices:
    train_test_features[index] = boxcox1p(train_test_features[index], boxcox_normmax(train_test_features[index] + 1))
for index in skew_indices:
    assessment(pd.concat([train_test_features.iloc[:len(train_targets), :], train_targets], axis=1), 'SalePrice', index, -1)
numerical_features_to_be_dropped = []
for feature in features_numerical:
    predominant_value_count = train_test_features[feature].value_counts().max()
    if predominant_value_count / train_test_features.shape[0] > 0.995:
        numerical_features_to_be_dropped.append(feature)
print(f'\nNumerical features to be dropped: {numerical_features_to_be_dropped}')
train_test_features = train_test_features.drop(numerical_features_to_be_dropped, axis=1)
features_numerical = []
for column in train_features.columns:
    if train_features[column].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
        features_numerical.append(column)

updated_train_set = pd.concat([train_test_features.iloc[:len(train_targets), :], train_targets], axis=1)

correlation_map(updated_train_set, 'SalePrice', 15)
missing_categorical = []
for feature in features_categorical:
    if train_test_features.count()[feature] < train_test_features.shape[0]:
        missing_categorical.append(feature)
print(f'\nCategorical features with missing values before treatment: {len(missing_categorical)}\n')
print(missing_categorical)
train_test_features['MSZoning'] = train_test_features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
train_test_features['Alley'] = train_test_features['Alley'].fillna('None')
train_test_features['Utilities'] = train_test_features['Utilities'].fillna('None')
train_test_features['Exterior1st'] = train_test_features['Exterior1st'].fillna(train_test_features['Exterior1st'].mode()[0])
train_test_features['Exterior2nd'] = train_test_features['Exterior2nd'].fillna(train_test_features['Exterior2nd'].mode()[0])
train_test_features['MasVnrType'] = train_test_features['MasVnrType'].fillna('None')
train_test_features['BsmtQual'] = train_test_features['BsmtQual'].fillna('None')
train_test_features['BsmtCond'] = train_test_features['BsmtCond'].fillna('None')
train_test_features['BsmtExposure'] = train_test_features['BsmtExposure'].fillna('None')
train_test_features['BsmtFinType1'] = train_test_features['BsmtFinType1'].fillna('None')
train_test_features['BsmtFinType2'] = train_test_features['BsmtFinType2'].fillna('None')
train_test_features['Electrical'] = train_test_features['Electrical'].fillna(train_test_features['Electrical'].mode()[0])
train_test_features['KitchenQual'] = train_test_features['KitchenQual'].fillna(train_test_features['KitchenQual'].mode()[0])
train_test_features['Functional'] = train_test_features['Functional'].fillna(train_test_features['Functional'].mode()[0])
train_test_features['FireplaceQu'] = train_test_features['FireplaceQu'].fillna('None')
train_test_features['GarageType'] = train_test_features['GarageType'].fillna('None')
train_test_features['GarageFinish'] = train_test_features['GarageFinish'].fillna('None')
train_test_features['GarageQual'] = train_test_features['GarageQual'].fillna('None')
train_test_features['GarageCond'] = train_test_features['GarageCond'].fillna('None')
train_test_features['PoolQC'] = train_test_features['PoolQC'].fillna('None')
train_test_features['Fence'] = train_test_features['Fence'].fillna('None')
train_test_features['MiscFeature'] = train_test_features['MiscFeature'].fillna('None')
train_test_features['SaleType'] = train_test_features['SaleType'].fillna(train_test_features['SaleType'].mode()[0])

missing_categorical.clear()
for feature in features_categorical:
    if train_test_features.count()[feature] < train_test_features.shape[0]:
        missing_categorical.append(feature)
print(f'\nCategorical features with missing values after treatment: {len(missing_categorical)}')
categorical_features_to_be_dropped = []
for feature in features_categorical:
    predominant_value_count = train_test_features[feature].value_counts().max()
    if predominant_value_count / train_test_features.shape[0] > 0.995:
        categorical_features_to_be_dropped.append(feature)
print(f'\nCategorical features to be dropped: {categorical_features_to_be_dropped}')
train_test_features = train_test_features.drop(categorical_features_to_be_dropped, axis=1)
map1 = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
set1 = ['ExterQual', 'ExterCond', 'BsmtQual','BsmtCond', 'HeatingQC',
        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
for feature in set1:
    train_test_features[feature] = train_test_features[feature].replace(map1)

map2 = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}
train_test_features['BsmtExposure'] = train_test_features['BsmtExposure'].replace(map2)

map3 = {'GLQ': 4,'ALQ': 3,'BLQ': 2,'Rec': 3,'LwQ': 2,'Unf': 1,'None': 0}
set3 = ['BsmtFinType1', 'BsmtFinType2']
for feature in set3:
    train_test_features[feature] = train_test_features[feature].replace(map3)

map4 = {'Y': 1, 'N': 0}
train_test_features['CentralAir'] = train_test_features['CentralAir'].replace(map4)

map5 = {'Typ': 3, 'Min1': 2.5, 'Min2': 2, 'Mod': 1.5, 'Maj1': 1, 'Maj2': 0.5, 'Sev': 0, 'Sal': 0}
train_test_features['Functional'] = train_test_features['Functional'].replace(map5)
train_test_features["TotalGarageQual"] = train_test_features["GarageQual"] * train_test_features["GarageCond"]
train_test_features["TotalExteriorQual"] = train_test_features["ExterQual"] * train_test_features["ExterCond"]
train_test_features = pd.get_dummies(train_test_features).reset_index(drop=True)
train_targets_skew = skew(train_targets)
print(f'Dependent variable skew factor: {train_targets_skew:.2f}')
y_train = np.log1p(train_targets)
fig, ax =plt.subplots(1, 2, figsize=(16,4))
chart1 = sns.distplot(train_targets, ax=ax[0], color='b')
chart1.set_xlabel('SalePrice',fontsize=12)
chart2 = sns.distplot(y_train, ax=ax[1], color='g')
chart2.set_xlabel('SalePrice',fontsize=12)
fig.show()
X_train = train_test_features.iloc[:len(train_targets), :]
X_test = train_test_features.iloc[len(train_targets):, :]
X_train_data = pd.concat([X_train, y_train], axis=1)
features_to_be_dropped = []
for feature in X_train.columns:
    all_value_counts = X_train[feature].value_counts()
    zero_value_counts = all_value_counts.iloc[0]
    if zero_value_counts / len(X_train) > 0.995:
        features_to_be_dropped.append(feature)
print('\nFeatures with predominant zeroes:\n')
print(features_to_be_dropped)

X_train = X_train.drop(features_to_be_dropped, axis=1).copy()
X_test = X_test.drop(features_to_be_dropped, axis=1).copy()
X_train_data = X_train_data.drop(features_to_be_dropped, axis=1).copy()
# lasso_alphas = [5e-5, 1e-4, 5e-4, 1e-3]
# lasso_outliers = make_pipeline(RobustScaler(),
#                       LassoCV(max_iter=1e7, alphas=lasso_alphas,
#                               random_state=42, cv=4))

# print(f'{"Identifying relevant outliers with LassoCV"}')
# score = xval_rmse_scoring(lasso_outliers, X_train, y_train, 4)
# print(f'{"All obs":<10}{score.mean():>14.8f}{score.std():>14.8f}')

# outlier_mean = []
# outlier_stdev = []

# for i in range(len(X_train)):
#     X_new = X_train.copy()
#     y_new = y_train.copy()
#     X_new = X_new.drop(X_new.index[i])
#     y_new = y_new.drop(y_new.index[i])
#     score = xval_rmse_scoring(lasso_outliers, X_new, y_new, 4)
#     outlier_mean.append(score.mean())
#     outlier_stdev.append(score.std())
#     print(f'{"Obs ":4}{i:<6}{score.mean():>14.8f}{score.std():>14.8f}')

# outlier_impact = pd.DataFrame(list(zip(outlier_mean, outlier_stdev)), columns =['Mean', 'St Dev'])
# outlier_lower_mean = outlier_impact.sort_values(by=['Mean'], ascending=True)
# outlier_lower_stdev = outlier_impact.sort_values(by=['St Dev'], ascending=True)
# outlier_impact.to_csv('outlier_impact.csv')
assessment(X_train_data, 'SalePrice', 'GrLivArea', 523)
assessment(X_train_data, 'SalePrice', 'GrLivArea', 1298)
outliers = [1298, 523, 30, 462, 588, 632, 1324]
X_train = X_train.drop(X_train.index[outliers])
y_train = y_train.drop(y_train.index[outliers])
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
elasticnet_alphas = [5e-5, 1e-4, 5e-4, 1e-3]
elasticnet_l1ratios = [0.8, 0.85, 0.9, 0.95, 1]
elasticnet = make_pipeline(RobustScaler(),
                           ElasticNetCV(max_iter=1e7, alphas=elasticnet_alphas,
                                        cv=kfolds, l1_ratio=elasticnet_l1ratios))

lasso_alphas = [5e-5, 1e-4, 5e-4, 1e-3]
lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas=lasso_alphas,
                              random_state=42, cv=kfolds))

ridge_alphas = [13.5, 14, 14.5, 15, 15.5]
ridge = make_pipeline(RobustScaler(),
                      RidgeCV(alphas=ridge_alphas, cv=kfolds))

gradb = GradientBoostingRegressor(n_estimators=6000, learning_rate=0.01,
                                  max_depth=4, max_features='sqrt',
                                  min_samples_leaf=15, min_samples_split=10,
                                  loss='huber', random_state=42)

svr = make_pipeline(RobustScaler(),
                    SVR(C=20, epsilon=0.008, gamma=0.0003))

xgboost = XGBRegressor(learning_rate=0.01, n_estimators=6000,
                       max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006, random_state=42)

stackcv = StackingCVRegressor(regressors=(elasticnet, gradb, lasso, 
                                          ridge, svr, xgboost),
                              meta_regressor=xgboost,
                              use_features_in_secondary=True)
print('Individual model scoring on cross-validation\n')
print(f'{"Model":<20}{"RMSE mean":>12}{"RMSE stdev":>12}\n')

score = xval_rmse_scoring(elasticnet, X_train, y_train, kfolds)
print(f'{"1. ElasticNetCV":<20}{score.mean():>12.4f}{score.std():>12.4f}')

score = xval_rmse_scoring(lasso, X_train, y_train, kfolds)
print(f'{"2. LassoCV":<20}{score.mean():>12.4f}{score.std():>12.4f}')

score = xval_rmse_scoring(ridge, X_train, y_train, kfolds)
print(f'{"3. RidgeCV":<20}{score.mean():>12.4f}{score.std():>12.4f}')

score = xval_rmse_scoring(gradb, X_train, y_train, kfolds)
print(f'{"4. GradientBoosting":<20}{score.mean():>12.4f}{score.std():>12.4f}')

score = xval_rmse_scoring(svr, X_train, y_train, kfolds)
print(f'{"5. SVR":<20}{score.mean():>12.4f}{score.std():>12.4f}')

score = xval_rmse_scoring(xgboost, X_train, y_train, kfolds)
print(f'{"6. XGBoost":<20}{score.mean():>12.4f}{score.std():>12.4f}')
print('\nFitting individual models to the training set\n')
print(f'{"1. ElasticNetCV...":<20}')
elastic_fit = elasticnet.fit(X_train, y_train)
print(f'{"2. LassoCV...":<20}')
lasso_fit = lasso.fit(X_train, y_train)
print(f'{"3. RidgeCV...":<20}')
ridge_fit = ridge.fit(X_train, y_train)
print(f'{"4. GradientBoosting...":<20}')
gradb_fit = gradb.fit(X_train, y_train)
print(f'{"5. SVR...":<20}')
svr_fit = svr.fit(X_train, y_train)
print(f'{"6. XGBoost...":<20}')
xgb_fit = xgboost.fit(X_train, y_train)

print('\nFitting the stacking model to the training set\n')
print(f'{"StackingCV...":<20}')
stackcv_fit = stackcv.fit(np.array(X_train), np.array(y_train))
blend_weights = [0.11, 0.05, 0.00, 0.14, 0.43, 0.00, 0.27]
y_train = np.expm1(y_train)
y_pred = np.expm1((blend_weights[0] * elastic_fit.predict(X_train)) +
                  (blend_weights[1] * lasso_fit.predict(X_train)) +
                  (blend_weights[2] * ridge_fit.predict(X_train)) +
                  (blend_weights[3] * svr_fit.predict(X_train)) +
                  (blend_weights[4] * gradb_fit.predict(X_train)) +
                  (blend_weights[5] * xgb_fit.predict(X_train)) +
                  (blend_weights[6] * stackcv_fit.predict(np.array(X_train))))
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
rmsle = np.sqrt(mean_squared_log_error(y_train, y_pred))
mae = mean_absolute_error(y_train, y_pred)
print('\nBlend model performance on the training set\n')
print(f'{"RMSE":<7} {rmse:>15.8f}')
print(f'{"RMSLE":<7} {rmsle:>15.8f}')
print(f'{"MAE":<7} {mae:>15.8f}')
# print('\nGenerating submission')
# submission = pd.read_csv('submission.csv')
# submission.iloc[:, 1] = np.round_(np.expm1((blend_weights[0] * elastic_fit.predict(X_test)) +
#                                            (blend_weights[1] * lasso_fit.predict(X_test)) +
#                                            (blend_weights[2] * ridge_fit.predict(X_test)) +
#                                            (blend_weights[3] * svr_fit.predict(X_test)) +
#                                            (blend_weights[4] * gradb_fit.predict(X_test)) +
#                                            (blend_weights[5] * xgb_fit.predict(X_test)) +
#                                            (blend_weights[6] * stackcv_fit.predict(np.array(X_test)))))
# submission.to_csv('../output/submission_new.csv', index=False)
# print('Submission saved')
end_time = datetime.now()

print('\nStart time', start_time)
print('End time', end_time)
print('Time elapsed', end_time - start_time)