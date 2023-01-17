import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import sklearn.model_selection as ms

import seaborn as sns



import sklearn.linear_model as lm



import sklearn.metrics as m



import sklearn.dummy as d

import sklearn.preprocessing as p



import sklearn.ensemble as e



import sklearn.pipeline as pipeline



import sklearn.svm as svm



import sklearn.impute as impute



from sklearn.compose import make_column_transformer



import scipy.stats as stats



import lightgbm as lgb

import xgboost as xgb
plt.figure(figsize=(3,3))
sns.set(rc={'figure.figsize':(20,15)})
hp = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
hp.head(20)
hp.shape
hp.columns
hp.skew()
hp.info()
def print_types_in_df(df):

    types = ['object', 'category', 'int64', 'float64']

    

    types_to_counts = {x: len(df.select_dtypes(include=[x]).columns.values) for x in types}

    

    print(types_to_counts)

    print('sum: '  +str(sum(types_to_counts.values())))
print_types_in_df(hp)
categorical_oh_features = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',

'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'Functional', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']



categorical_l_features = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']



numerical_features = ['LotFrontage', 'LotArea',  'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']



date_features = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']



unused_features = ['Id']



target_feature = ['SalePrice']
len(categorical_oh_features) + len(categorical_l_features) + len(numerical_features) + len(date_features) + len(unused_features) + len(target_feature)
hp[categorical_oh_features] = hp[categorical_oh_features].astype('category')
hp[categorical_l_features] = hp[categorical_l_features].astype('category')
hp[numerical_features] = hp[numerical_features].apply(pd.to_numeric, errors='coerce')
print_types_in_df(hp)
hp.drop(columns=unused_features, inplace=True)
hp['SalePrice'].describe()
sns.boxplot(hp['SalePrice'],              

            showmeans=True,

            medianprops={'color': 'r'})
discrete_cols=[]

for col in hp.columns:

    if hp[col].dtype=='int64' and len(hp[col].unique()) <=15:

        discrete_cols.append(col)

        

discrete_cols
for c in discrete_cols:

    print(f'{c}:', end='')

    print(hp[c].unique())
len(discrete_cols)
for i in range(len(discrete_cols)):

    plt.subplot(4,3, i+1)

    sns.boxplot(data=hp[discrete_cols[i]], 

                showmeans=True,

                medianprops={'color': 'r'})

    plt.title(discrete_cols[i])
for i in range(len(discrete_cols)):

    plt.subplot(4,3, i+1)

    sns.barplot(data=hp, 

                x=discrete_cols[i],

               y='SalePrice')
continuous_cols=[]

for col in hp.columns:

    if (hp[col].dtype=='int' or hp[col].dtype=='float') and col not in ['Id', 'YearBuilt','YearRemodAdd','GarageYrBlt'] and col not in discrete_cols:

        continuous_cols.append(col)

        

continuous_cols
len(continuous_cols)
corr = hp[continuous_cols].corr()

corr
sns.heatmap(corr, annot=True, cmap='coolwarm')
t = abs(corr.SalePrice).sort_values(ascending=False)



print(t)

t.index
features_to_eda = ['GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']



for i in range(len(features_to_eda)):

    plt.subplot(2,2, i+1)

    sns.regplot(x=hp[features_to_eda[i]], y=hp.SalePrice)
categorical_cols = hp.select_dtypes(include=['category']).columns.values

categorical_cols
cat1 = ['MSSubClass']
sns.boxplot(data=hp, x='MSSubClass', y='SalePrice')
pt = hp.pivot_table(index='MSSubClass', values='SalePrice', aggfunc=np.median)
pt.sort_values(by='SalePrice',ascending=False)
cat2=['MSZoning','Street','LotShape','LandContour','LotConfig','LandSlope']
for i in range(len(cat2)):

    plt.subplot(2,3, i+1)

    sns.boxplot(data=hp, 

                x=cat2[i],

               y='SalePrice')
year_features = ['YearBuilt','YearRemodAdd','YrSold','GarageYrBlt']
for i in range(len(year_features)):

    plt.subplot(2,2, i+1)

    sns.lineplot(x=year_features[i], y='SalePrice', data=hp, estimator=np.median)
for i in range(len(year_features)):

    plt.subplot(2,2, i+1)

    sns.boxplot(x=year_features[i], y='SalePrice', data=hp)
t = pd.DataFrame()
t['SalePrice'] = hp.SalePrice

t['YrSold'] = hp.YrSold

t['YearBuilt'] = hp.YrSold - hp.YearBuilt

t['YearRemodAdd'] = hp.YrSold - hp.YearRemodAdd

t['GarageYrBlt'] = hp.YrSold - hp.GarageYrBlt
for i in range(len(year_features)):

    plt.subplot(2,2, i+1)

    sns.boxplot(x=year_features[i], y='SalePrice', data=t)
nulls = hp.isna().sum().sort_values(ascending=False)



nulls_df = pd.DataFrame({'feature': nulls.index, 'value': nulls.values})

nulls_df
sns.factorplot('value', 'feature', data=nulls_df[nulls_df['value'] > 0], kind='bar')
nulls_with_type = pd.concat([hp[nulls_df.feature.values].dtypes, nulls], axis=1, keys=['type', 'count_of_nulls'])

nulls_with_type[nulls_with_type.count_of_nulls > 0]
nulls_with_type.index
numericals_nulls = nulls_with_type[nulls_with_type.index.isin(numerical_features)]

numericals_nulls
numericals_nulls_above_zero = numericals_nulls[numericals_nulls.count_of_nulls > 0]

for i in range(len(numericals_nulls_above_zero.index.values)):

    plt.subplot(1,2, i+1)

    sns.boxplot(data=hp[numericals_nulls_above_zero.index.values[i]], 

                showmeans=True,

                medianprops={'color': 'r'})

    plt.title(numericals_nulls_above_zero.index.values[i])
hp[numerical_features].isnull().sum()
numerical_imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')

hp[numerical_features] = numerical_imputer.fit_transform(hp[numerical_features])
hp[numerical_features].isnull().sum().sum()
date_nulls = nulls_with_type[nulls_with_type.index.isin(date_features)]

date_nulls
date_nulls_above_zero = date_nulls[date_nulls.count_of_nulls > 0]

date_nulls_above_zero
sns.boxplot(data=hp[date_nulls_above_zero.index.values[0]], 

                showmeans=True,

                medianprops={'color': 'r'})
hp.GarageYrBlt.median()
date_imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')

hp[date_features] = date_imputer.fit_transform(hp[date_features])
hp[date_features].isnull().sum().sum()
categorical_features_to_impute_none = []
_features_to_drop = ['PoolQC','MiscFeature','Alley','Fence']
unused_features.extend(_features_to_drop)
categorical_oh_features = list(set(categorical_oh_features).difference(set(_features_to_drop)))
categorical_l_features = list(set(categorical_l_features).difference(set(_features_to_drop)))
hp.drop(columns=_features_to_drop,axis=1,inplace=True)
categorical_oh_nulls = nulls_with_type[nulls_with_type.index.isin(categorical_oh_features)]

categorical_oh_nulls
categorical_l_nulls = nulls_with_type[nulls_with_type.index.isin(categorical_l_features)]

categorical_l_nulls
hp['FireplaceQu'].unique()
hp[hp['Fireplaces'] == 0]['FireplaceQu'].unique()
# FireplaceQu nulls to ImputedNone

categorical_features_to_impute_none += ['FireplaceQu']
hp['MasVnrType'].unique()
hp[hp['MasVnrArea'] == 0]['MasVnrType'].unique()
hp[hp['MasVnrType'].isna()]['MasVnrArea'].unique()
# MasVnrType nulls to ImputedNone

categorical_features_to_impute_none += ['MasVnrType']
bsmt_cols = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

 'BsmtFinType2']



for c in bsmt_cols:

    print(f'{c}:', end='')

    print(hp[c].unique())
hp[hp['BsmtQual'].isna()][['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1',

                        'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']]
hp[bsmt_cols].mode()
hp[bsmt_cols].isna().sum()
# bsmt_cols nulls to ImputedNone

categorical_features_to_impute_none += bsmt_cols
hp[hp['Electrical'].isna()]
hp['Electrical'].mode()
# Electrical nulls to mode
garage_cols = ['GarageType','GarageFinish','GarageQual','GarageCond']





hp[hp['GarageType'].isna()][garage_cols]
hp[garage_cols].isna().sum()
# garage_cols to ImputedNone

categorical_features_to_impute_none += garage_cols
categorical_features_to_impute_none
categorical_features_to_impute_mode = list(set(categorical_oh_features + categorical_l_features).difference(set(categorical_features_to_impute_none)))

categorical_features_to_impute_mode
categorical_none_imputer = impute.SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='ImputedNone')

hp[categorical_features_to_impute_none] = categorical_none_imputer.fit_transform(hp[categorical_features_to_impute_none])
categorical_mode_imputer = impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent')

hp[categorical_features_to_impute_mode] = categorical_mode_imputer.fit_transform(hp[categorical_features_to_impute_mode])
hp.isna().sum().sum()
sns.distplot(hp['SalePrice'])
stats.probplot(hp.SalePrice, plot=sns.mpl.pyplot)
sns.residplot(hp.GrLivArea, hp.SalePrice)
hp.SalePrice.skew(), hp.SalePrice.kurtosis()
print_types_in_df(hp)
continuous=[]

for col in hp.columns:

    if (hp[col].dtype == 'int64' or hp[col].dtype == 'float64') and col != 'Id' and len(hp[col].unique()) >15 and col not in date_features:

        continuous.append(col)

        

continuous
hp.LotFrontage.skew()
skewed_features=[]

for col in continuous:

    if hp[col].skew()>0 or hp[col].skew()<0:

        skewed_features.append(col)
skewed_features
apply_log=[]

for col in skewed_features:

    if 0 not in hp[col].unique():

        apply_log.append(col)

apply_log
hp[apply_log].skew()
hp[apply_log] = np.log(hp[apply_log])
hp[apply_log].skew()
for i in range(len(apply_log)):

    plt.subplot(3,2, i+1)

    sns.distplot(hp[apply_log[i]])
stats.probplot(hp.SalePrice, plot=sns.mpl.pyplot)
def bigger_than_zero(val):

    return 1 if val > 0 else 0
def create_features(df):

    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

        

    df['TotalFeet'] = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF']

    

    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])

    

    df['TotalPorch'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']

    

    df['HasPool'] = df['PoolArea'].apply(bigger_than_zero)

    df['Has2ndFlr'] = df['2ndFlrSF'].apply(bigger_than_zero)

    df['HasGarage'] = df['GarageArea'].apply(bigger_than_zero)

    df['HasBsmt'] = df['TotalBsmtSF'].apply(bigger_than_zero)

    df['HasFireplace'] = df['Fireplaces'].apply(bigger_than_zero)

create_features(hp)
one_hot_encoder = p.OneHotEncoder(sparse=False, handle_unknown='ignore')

hp_oh = pd.DataFrame(one_hot_encoder.fit_transform(hp[categorical_oh_features]), columns=one_hot_encoder.get_feature_names(categorical_oh_features))

hp_oh
hp_oh.index = hp.index



hp.drop(categorical_oh_features, axis=1, inplace=True)



hp = pd.concat([hp, hp_oh], axis=1)



hp
ordinal_encoder = p.OrdinalEncoder()

hp[categorical_l_features] = pd.DataFrame(ordinal_encoder.fit_transform(hp[categorical_l_features]), columns=hp[categorical_l_features].columns)

hp[categorical_l_features]
hp.isna().sum().sum()
isolation_forest = e.IsolationForest()

isolation_forest.fit(hp)

outliers_res = isolation_forest.predict(hp)
outliers = hp.iloc[outliers_res == -1].index
t = hp.drop(outliers, axis=0)

t.isna().sum().sum()

hp = t
hp.isna().sum().sum()
hp = hp.sample(frac=1).reset_index(drop=True)
y = hp.SalePrice
X = hp.drop(columns='SalePrice')
X_train, X_hold_out, X_val  = np.split(X, [int(.7*len(X)), int(.9*len(X))])



y_train, y_hold_out, y_val  = np.split(y, [int(.7*len(y)), int(.9*len(y))])
hp.isna().sum().sum()
# X_train, X_val, y_train, y_val = ms.train_test_split(X, y, test_size=0.2)
scaler = p.StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)

X_hold_out = pd.DataFrame(scaler.transform(X_hold_out),columns=X_hold_out.columns)

X_val = pd.DataFrame(scaler.transform(X_val),columns=X_val.columns)



X = pd.DataFrame(scaler.transform(X),columns=X.columns)

X
# parameters = {

#     'C': [1, 1.5, 2, 5],

#     'epsilon': [0.001, 0.01, 0.1],

#     'kernel': ['linear', 'poly', 'rbf'],

#     'gamma': [0.001, 0.01, 0.1, 1]

# }



# grid_search = ms.GridSearchCV(svm.SVR(), parameters, n_jobs=-1, cv=10, return_train_score=True, verbose=20)

# grid_search.fit(X, y)

# grid_search.best_params_, grid_search.best_score_
# svr_results = pd.DataFrame(grid_search.cv_results_)

# svr_results.sort_values(by='rank_test_score', inplace=True)

# svr_results.columns

# svr_results[['param_C', 'param_epsilon', 'param_gamma', 'param_kernel', 'rank_test_score', 'mean_test_score']].head(30)
# parameters = {

#     'C': [0.5, 0.75, 1, 1.25, 1.5, 2, 4, 5],

#     'epsilon': [0.0001, 0.0005, 0.001, 0.01, 0.1],

#     'kernel': ['rbf'],

#     'gamma': [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]

# }



# grid_search = ms.GridSearchCV(svm.SVR(), parameters, n_jobs=-1, cv=10, return_train_score=True, verbose=5)

# grid_search.fit(X, y)

# grid_search.best_params_, grid_search.best_score_
# svr_results2 = pd.DataFrame(grid_search.cv_results_)

# svr_results2.sort_values(by='rank_test_score', inplace=True)

# svr_results2.columns

# svr_results2[['param_C', 'param_epsilon', 'param_gamma', 'param_kernel', 'rank_test_score', 'mean_test_score']].head(30)
# parameters = {

#     'alpha': [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1],

#     'l1_ratio': [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1],

# }



# grid_search = ms.GridSearchCV(lm.ElasticNet(), parameters, n_jobs=-1, cv=10, return_train_score=True, verbose=5)

# grid_search.fit(X, y)

# grid_search.best_params_, grid_search.best_score_
# grid_search.best_params_, grid_search.best_score_
# parameters = {

#     'learning_rate': [0.01, 0.05, 0.1, 0.5],

#     'n_estimators': [1200, 1300, 1500, 1700],

#     'max_depth': [2, 3, 4]

# }



# grid_search = ms.GridSearchCV(e.GradientBoostingRegressor(), parameters, n_jobs=-1, cv=10, return_train_score=True, verbose=20)

# grid_search.fit(X, y)

# grid_search.best_params_, grid_search.best_score_
# grid_search.best_params_, grid_search.best_score_
# parameters = {

#     'learning_rate': [0.001, 0.01],

#     'n_estimators': [1500, 2000, 2500],

#     'max_depth': [-1, 3, 5, 10, 20],

#     'num_leaves': [5, 20, 30],

#     'min_data_in_leaf': [15, 20],

#     'lambda_l1': [0, 0.1],

#     'lambda_l2': [0, 0.1]

# }



# grid_search = ms.GridSearchCV(lgb.LGBMRegressor(), parameters, n_jobs=-1, cv=10, return_train_score=True, verbose=8)

# grid_search.fit(X_train, y_train)

# grid_search.best_params_, grid_search.best_score_
# grid_search.best_params_, grid_search.best_score_
# lgbmr_results = pd.DataFrame(grid_search.cv_results_)

# lgbmr_results.sort_values(by='rank_test_score', inplace=True)

# lgbmr_results.columns

# lgbmr_results[['param_lambda_l1', 'param_lambda_l2', 'param_learning_rate',

#        'param_max_depth', 'param_min_data_in_leaf', 'param_n_estimators',

#        'param_num_leaves', 'rank_test_score', 'mean_test_score']].head(10)
# parameters = {

#     'learning_rate': [0.005, 0.01, 0.05],

#     'n_estimators': [2000, 2500, 3000],

# #     'max_depth': [-1, 3, 5, 10, 20],

#     'num_leaves': [3, 4, 5, 6],

#     'min_data_in_leaf': [15, 20, 25],

#     'lambda_l1': [0.001, 0.01, 0.1],

#     'lambda_l2': [0.001, 0.01, 0.1]

# }



# grid_search = ms.GridSearchCV(lgb.LGBMRegressor(), parameters, n_jobs=-1, cv=10, return_train_score=True, verbose=8)

# grid_search.fit(X_train, y_train)

# grid_search.best_params_, grid_search.best_score_
# lgbmr_results2 = pd.DataFrame(grid_search.cv_results_)

# lgbmr_results2.sort_values(by='rank_test_score', inplace=True)

# lgbmr_results2.columns

# lgbmr_results2[['param_lambda_l1', 'param_lambda_l2', 'param_learning_rate',

#        'param_min_data_in_leaf', 'param_n_estimators',

#        'param_num_leaves', 'rank_test_score', 'mean_test_score']].head(10)
# parameters = {

#     'learning_rate': [0.01, 0.1, 0.001],

#     'n_estimators': [1700, 1800, 2000, 2200, 2400],

#     'max_depth': [3, 4, 5],

#     'num_leaves': [3, 4, 6]

# }



# grid_search = ms.GridSearchCV(xgb.XGBRegressor(), parameters, n_jobs=-1, cv=ps, return_train_score=True, verbose=10)

# grid_search.best_params_, grid_search.best_score_
# r2_scores = ms.cross_val_score(xgb.XGBRegressor(learning_rate=0.01, max_depth=4, n_estimators=2400), X=X, y=y, cv=10)

# r2_scores.mean(), np.median(r2_scores) 
# r2_scores = ms.cross_val_score(xgb.XGBRegressor(learning_rate=0.01, max_depth=4, n_estimators=3000), X=X, y=y, cv=10)

# r2_scores.mean(), np.median(r2_scores) 
# r2_scores = ms.cross_val_score(xgb.XGBRegressor(learning_rate=0.01, max_depth=4, n_estimators=4000), X=X, y=y, cv=10)

# r2_scores.mean(), np.median(r2_scores) 
regressor_svr = svm.SVR(kernel='rbf', C=2, epsilon=0.01, gamma=0.0005) 

regressor_en = lm.ElasticNet(alpha=0.1, l1_ratio=0.025)

regressor_gbr = e.GradientBoostingRegressor(learning_rate=0.05, max_depth=3, n_estimators=1200)

regressor_lgbm = lgb.LGBMRegressor(lambda_l1=0.01, lambda_l2=0.1, learning_rate=0.05,

              min_data_in_leaf=25, n_estimators=2000, num_leaves=4)

regressor_xgb = xgb.XGBRegressor(learning_rate=0.01, max_depth=4, n_estimators=4000)



ensemble_of_regressors = [regressor_svr, regressor_en, regressor_gbr, regressor_lgbm, regressor_xgb]
for clf in ensemble_of_regressors:

    clf.fit(X_train, y_train)
def get_predictions(x, ensemble_of_regressors, y=None):

    pred_result = pd.DataFrame()

    

    i = 1

    for clf in ensemble_of_regressors:

        y_pred = clf.predict(x)

        if y is not None:

            print(clf.__class__.__name__, m.r2_score(y, y_pred))

        

        pred_result.insert(i - 1, 'y_pred_' + str(i), y_pred)

        i += 1

        

    return pred_result
pred_result = get_predictions(X_hold_out, ensemble_of_regressors, y_hold_out)
X_stack_train = pred_result



y_stack_train = y_hold_out
blender_of_ensamble = lm.LinearRegression()



blender_of_ensamble.fit(X_stack_train, y_stack_train)
pred_result_test = get_predictions(X_val, ensemble_of_regressors, y_val)
X_stack_test = pred_result_test



y_stack_pred = blender_of_ensamble.predict(X_stack_test)

X_stack_test.shape, y_stack_pred.shape
m.r2_score(y_stack_pred, y_val)
estimators2 = [

('svr', svm.SVR(kernel='rbf', C=2, epsilon=0.01, gamma=0.0005)),

('en', lm.ElasticNet(alpha=0.1, l1_ratio=0.03)),

('gbr', e.GradientBoostingRegressor(learning_rate=0.05, max_depth=3, n_estimators=1200)),

('lgbm', lgb.LGBMRegressor(lambda_l1=0.01, lambda_l2=0.1, learning_rate=0.05,

              min_data_in_leaf=25, n_estimators=2000, num_leaves=4)),

('xgb', xgb.XGBRegressor(learning_rate=0.01, max_depth=4, n_estimators=4000))

]
s_reg = e.StackingRegressor(

    estimators=estimators2,

    cv=20,

    n_jobs=-1

)
X_train, X_val, y_train, y_val = ms.train_test_split(X, y, test_size=0.2)
s_reg.fit(X_train, y_train)
s_reg.score(X_val, y_val)
s_reg.fit(X, y)
hp_pred = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
X_pred = hp_pred.drop(columns=unused_features)
X_pred.shape
X_pred.isna().sum().sum()
X_pred[numerical_features] = numerical_imputer.transform(X_pred[numerical_features])
X_pred[date_features] = date_imputer.transform(X_pred[date_features])
X_pred[categorical_features_to_impute_mode] = categorical_mode_imputer.transform(X_pred[categorical_features_to_impute_mode])



X_pred[categorical_features_to_impute_none] = categorical_none_imputer.transform(X_pred[categorical_features_to_impute_none])
X_pred[['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']] = np.log(X_pred[['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']])
create_features(X_pred)
X_pred.shape
X_pred_oh = pd.DataFrame(one_hot_encoder.transform(X_pred[categorical_oh_features]), columns=one_hot_encoder.get_feature_names(categorical_oh_features))

X_pred_oh
X_pred_oh.index = X_pred.index



X_pred.drop(categorical_oh_features, axis=1, inplace=True)



X_pred = pd.concat([X_pred, X_pred_oh], axis=1)



X_pred
X_pred[categorical_l_features] = pd.DataFrame(ordinal_encoder.transform(X_pred[categorical_l_features]), columns=X_pred[categorical_l_features].columns)

X_pred[categorical_l_features]
X_pred = pd.DataFrame(scaler.transform(X_pred),columns=X_pred.columns)

X_pred
# X_pred.shape
# pred_result = get_predictions(X_pred, ensemble_of_regressors)
# pred_result.shape
# X_pred_stack = pred_result



# y_pred_stack = blender_of_ensamble.predict(X_pred_stack)
# y_pred_stack.shape
y_pred = pd.DataFrame()

y_pred['Id'] = hp_pred.Id
y_pred['SalePrice'] = np.exp(s_reg.predict(X_pred))
# y_pred['SalePrice'] = np.exp(lgbm.predict(X_pred))
# y_pred['SalePrice'] = np.exp(y_pred_stack)
# y_pred['SalePrice'] = np.exp(regressor_en.predict(X_pred))
y_pred.shape
y_pred.to_csv('submission.csv', index=False)