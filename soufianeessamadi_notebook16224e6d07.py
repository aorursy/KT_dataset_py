import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import lightgbm as lgb



from sklearn.svm import SVR

from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from mlxtend.regressor import StackingCVRegressor



from scipy.stats import norm, skew

from sklearn.model_selection import KFold, cross_val_score

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline
train_df, test_df = None, None



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file_abs_path = os.path.join(dirname, filename)

        print(file_abs_path)

        if filename == 'train.csv':

            train_df = pd.read_csv(file_abs_path)

            print('train data loaded.')

        elif filename == 'test.csv':

            test_df = pd.read_csv(file_abs_path)

            print('test data loaded')
column_types_df = pd.DataFrame(train_df.dtypes, index=train_df.columns, columns=['type'])

column_types_df['type'].value_counts()
sns.set()

sns.distplot(train_df['SalePrice'] , fit=norm)

plt.ylabel('Frequency')

plt.title('SalePrice distribution')
print('Skewness: ', train_df['SalePrice'].skew(), '- Kurtosis: ', train_df['SalePrice'].kurt())
train_df['SalePrice'].describe()
numeric_cols = train_df.select_dtypes(include=['number'])

numeric_cols.columns
num_cols_to_plot = numeric_cols.columns.values[1:-1]



plt.figure(figsize=(20, 55))



for col_idx, col_name in enumerate(num_cols_to_plot):

    plt.subplot(numeric_cols.shape[1] / 3, 3, col_idx+1)

    sns.scatterplot(x=col_name, y='SalePrice', data=numeric_cols)

    plt.xlabel(col_name, fontsize=16)

    plt.ylabel('SalePrice', fontsize=16)



plt.tight_layout()        

plt.show()
categorical_cols = train_df.select_dtypes(include=['object'])

categorical_cols.columns
cat_cols_to_plot = categorical_cols.columns.values

categorical_cols_with_target = pd.concat([categorical_cols, train_df['SalePrice']], axis=1)



plt.figure(figsize=(25, 60))



for col_idx, col_name in enumerate(cat_cols_to_plot):

    plt.subplot((categorical_cols.shape[1] / 3) + 1, 3, col_idx+1)

    sns.boxplot(x=col_name, y='SalePrice', data=categorical_cols_with_target)

    plt.xlabel(col_name, fontsize=16)

    plt.ylabel('SalePrice', fontsize=16)

    plt.xticks(rotation=90)



plt.tight_layout()        

plt.show()
# Correlation map to see how features are correlated, excluding 'Id' col

corr_mat = train_df.iloc[:, 1:].corr()

plt.subplots(figsize=(16,16))

sns.heatmap(corr_mat, vmax=0.99, square=True)
most_corr_with_target = corr_mat['SalePrice'].nlargest(10).index

plt.subplots(figsize=(10,10))

sns.heatmap(train_df[most_corr_with_target].corr(), annot=True, square=True)
lines_to_remove_index = train_df['GrLivArea'][(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 400000)].index.values

train_df.drop(lines_to_remove_index, inplace=True)
train_df['row_usage'] = 'train'

test_df['row_usage'] = 'test'



merged_df = pd.concat((train_df, test_df)).reset_index(drop=True)
na_cols_percentage_ser = (merged_df.isna().sum()/merged_df.shape[0]) * 100

na_cols_percentage_ser[na_cols_percentage_ser > 0].sort_values(ascending=False)
cat_cols_with_na = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MasVnrType']

num_cols_with_na = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']



merged_df[cat_cols_with_na] = merged_df[cat_cols_with_na].fillna('none')

merged_df[num_cols_with_na] = merged_df[num_cols_with_na].fillna(0)

merged_df['Functional'].fillna('Typ', inplace=True)
merged_df.loc[merged_df['row_usage'] == 'train', 'LotFrontage'] = merged_df[merged_df['row_usage'] == 'train'].groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

merged_df.loc[merged_df['row_usage'] == 'test', 'LotFrontage'] = merged_df[merged_df['row_usage'] == 'test'].groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# replace Electrical NaN

merged_df['Electrical'].fillna(merged_df[merged_df['row_usage'] == 'train']['Electrical'].mode()[0], inplace=True)



# replace MSZoning NaN

merged_df['MSZoning'].fillna(merged_df[merged_df['row_usage'] == 'test']['MSZoning'].mode()[0], inplace=True)



# replace Utilities NaN

merged_df['Utilities'].fillna(merged_df[merged_df['row_usage'] == 'test']['Utilities'].mode()[0], inplace=True)



# replace Exterior1st NaN

merged_df['Exterior1st'].fillna(merged_df[merged_df['row_usage'] == 'test']['Exterior1st'].mode()[0], inplace=True)



# replace Exterior2nd NaN

merged_df['Exterior2nd'].fillna(merged_df[merged_df['row_usage'] == 'test']['Exterior2nd'].mode()[0], inplace=True)



# replace KitchenQual NaN

merged_df['KitchenQual'].fillna(merged_df[merged_df['row_usage'] == 'test']['KitchenQual'].mode()[0], inplace=True)



# replace SaleType NaN

merged_df['SaleType'].fillna(merged_df[merged_df['row_usage'] == 'test']['SaleType'].mode()[0], inplace=True)
cols_na_count = merged_df.isna().sum()

cols_na_count[cols_na_count > 0].sort_values(ascending=False)
merged_df.loc[merged_df['YearRemodAdd'] < merged_df['YearBuilt'], 'YearRemodAdd'] = merged_df.loc[merged_df['YearRemodAdd'] < merged_df['YearBuilt'], 'YearBuilt']
merged_df.loc[merged_df['YearRemodAdd'] > merged_df['YrSold'], 'YrSold'] = merged_df.loc[merged_df['YearRemodAdd'] > merged_df['YrSold'], 'YearRemodAdd']
# Total sqfootage measure

merged_df['TotalSF'] = merged_df['TotalBsmtSF'] + merged_df['1stFlrSF'] + merged_df['2ndFlrSF']

# Total bathrooms

merged_df['TotalBath'] = merged_df['BsmtFullBath'] + merged_df['FullBath']

# Total half bathrooms

merged_df['TotalHalfBath'] = merged_df['BsmtHalfBath'] + merged_df['HalfBath']

# Total porch sqfootage measure

merged_df['TotalPorchSF'] = merged_df['OpenPorchSF'] + merged_df['3SsnPorch'] + merged_df['EnclosedPorch'] + merged_df['ScreenPorch']

# Property age at sale

merged_df['AgeAtSale'] = merged_df['YrSold'] - merged_df['YearBuilt']

# Remodel age

merged_df['AgeAtRemodel'] = merged_df['YrSold'] - merged_df['YearBuilt']

# Years since remodel

merged_df['YearsSinceRemodel'] = merged_df['YearRemodAdd'] - merged_df['YearBuilt']
numeric_cols = merged_df.select_dtypes(include=['number'])

skewed_features_ser = numeric_cols.iloc[:,1:].apply(lambda x: skew(x)).sort_values(ascending=False)

skewed_features_ser
skewed_features_list = skewed_features_ser[skewed_features_ser > 1].index.values

skewed_features_list
skew(merged_df.loc[merged_df['row_usage'] == 'train', 'SalePrice'])
merged_df[skewed_features_list] = np.log1p(merged_df[skewed_features_list])



merged_df.loc[merged_df['row_usage'] == 'train', 'SalePrice'] = merged_df.loc[merged_df['row_usage'] == 'train', 'SalePrice'].apply(np.log)
merged_df.select_dtypes(include=['number']).iloc[:,1:].apply(lambda x: skew(x)).sort_values(ascending=False)

print(skew(merged_df.loc[merged_df['row_usage'] == 'train', 'SalePrice']))
ordinal_cat_features = ['BsmtFinType1', 'BsmtFinType2', 'BsmtExposure', 'Utilities', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'GarageFinish', 'PoolQC', 'CentralAir', 'LotShape', 'Street', 'Alley', 'PavedDrive']



common_encoding_dict = {

    'Ex'    : 5,

    'Gd'    : 4,

    'TA'    : 3,

    'Fa'    : 2,

    'Po'    : 1,

    'NA'    : 0,

    'Av'    : 3,

    'Mn'    : 2,

    'No'    : 1,

    'Fin'   : 3,

    'RFn'   : 2,

    'Unf'   : 1,

    'AllPub': 4,

    'NoSewr': 3,

    'NoSeWa': 2,

    'ELO'   : 1,

    'GLQ'   : 6,

    'ALQ'   : 5,

    'BLQ'   : 4,

    'Rec'   : 3,

    'LwQ'   : 2,

    'N'     : 0,

    'Y'     : 2,

    'Reg'   : 3,

    'IR1'	: 2,

    'IR2'	: 1,

    'IR3'	: 0,

    'Grvl'	: 1,

    'Pave'	: 2,

    'P'     : 1,

    'none'  : 0

}



merged_df[ordinal_cat_features] = merged_df[ordinal_cat_features].replace(common_encoding_dict)

merged_df['OverallScore'] = merged_df['OverallQual'] * merged_df['OverallCond']

merged_df['GarageScore'] = merged_df['GarageQual'] * merged_df['GarageCond']

merged_df['ExterScore'] = merged_df['ExterQual'] * merged_df['ExterCond']
column_types_df = pd.DataFrame(merged_df.dtypes, index=merged_df.columns, columns=['type'])

column_types_df['type'].value_counts()
print('merged dataframe shape before transforming categorical cols : ', merged_df.shape)



row_usage_ser = merged_df['row_usage']

merged_df = pd.get_dummies(merged_df.drop(columns=['row_usage']))

merged_df['row_usage'] = row_usage_ser



print('merged dataframe shape after transforming categorical cols : ', merged_df.shape)
processed_train_df = merged_df[merged_df['row_usage'] == 'train'].drop(columns=['Id', 'row_usage', 'SalePrice'])

target = merged_df[merged_df['row_usage'] == 'train']['SalePrice']

processed_test_df = merged_df[merged_df['row_usage'] == 'test'].drop(columns=['Id', 'row_usage', 'SalePrice'])

print('train_df shape: ', processed_train_df.shape, '- test_df shape: ', processed_test_df.shape, '- target shape: ', target.shape)

# Let's define a class to hold score evaluation stuff

class ScoreEvaluator:



    def __init__(self, folds_nb):

        self.k_fold = KFold(n_splits=folds_nb, shuffle=True, random_state=18)



    def eval_rmse_cv(self, model, X_train, y_train):

        return np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = self.k_fold))

# Linear regressor

lin_reg = make_pipeline(RobustScaler(), LinearRegression())



alphas = [.0001, .0003, .0005, .0007, .0009, 

          .01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 30]

l1_ratios = [0.8, 0.85, 0.9, 0.95, 0.99, 1]



# Lasso regressor

# lasso = make_pipeline(RobustScaler(), LassoCV(alphas=alphas, random_state=18))

lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0007, random_state=18))





# Ridge regressor

# ridge = RidgeCV(alphas=alphas)

ridge = Ridge(alpha=10)



# ElasticNet

# elastic_net = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios)

elastic_net = ElasticNet(alpha=0.0007, l1_ratio=0.3)





# Support Vector Regressor

svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon= 0.008, gamma=0.0003))



# LightGBM Regressor

light_gbm = lgb.LGBMRegressor(objective='regression',num_leaves=6,

                              learning_rate=0.01, n_estimators=7000,

                              max_bin=200, bagging_fraction=0.8,

                              bagging_freq=4, bagging_seed=18,

                              feature_fraction=0.2,

                              feature_fraction_seed=18,

                              min_sum_hessian_in_leaf=11,

                              random_state=18)



# XGBoost Regressor

xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3500,

                       max_depth=3, gamma=0, subsample=0.7,

                       objective='reg:squarederror',

                       colsample_bytree=0.7, n_jobs=-1,

                       scale_pos_weight=1, seed=18,

                       reg_alpha=0.00006, min_child_weight=0,

                       random_state=18)
lin_reg_rmse = ScoreEvaluator(10).eval_rmse_cv(lin_reg, processed_train_df.copy(), target.copy()).mean()

lasso_mean_rmse = ScoreEvaluator(10).eval_rmse_cv(lasso, processed_train_df.copy(), target.copy()).mean()

ridge_mean_rmse = ScoreEvaluator(10).eval_rmse_cv(ridge, processed_train_df.copy(), target.copy()).mean()

elastic_net_mean_rmse = ScoreEvaluator(10).eval_rmse_cv(elastic_net, processed_train_df.copy(), target.copy()).mean()

light_gbm_mean_rmse = ScoreEvaluator(10).eval_rmse_cv(light_gbm, processed_train_df.copy(), target.copy()).mean()

svr_mean_rmse = ScoreEvaluator(10).eval_rmse_cv(svr, processed_train_df.copy(), target.copy()).mean()

xgboost_mean_rmse = ScoreEvaluator(10).eval_rmse_cv(xgboost, processed_train_df.copy(), target.copy()).mean()
print('lin_reg rmse score: ', lin_reg_rmse)

print('lasso rmse score: ', lasso_mean_rmse)

print('ridge rmse score: ', ridge_mean_rmse)

print('elastic net rmse score: ', elastic_net_mean_rmse)

print('SVR rmse score: ', svr_mean_rmse)

print('LGBM rmse score: ', light_gbm_mean_rmse)

print('XGBoost rmse score: ', xgboost_mean_rmse)
# StackingCVRegressor 

stacked_reg = StackingCVRegressor(regressors=[ridge, svr, light_gbm, xgboost],

                                  meta_regressor=light_gbm,

                                  use_features_in_secondary=True)



stacked_mean_rmse = ScoreEvaluator(10).eval_rmse_cv(stacked_reg, processed_train_df.values, target.values).mean()

print('stacked reg score: ', stacked_mean_rmse)

# Class for handling combined prediction stuff



class CombinedModelsPredictor:



    models_to_combine_list = [svr, elastic_net, ridge, lasso, xgboost, stacked_reg, light_gbm]

    models_weights_list = [0.25, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1]



    def __init__(self, X_train, y_train):

        self.X_train = X_train

        self.y_train = y_train



    def fit_and_predict_weighted_model(self, model, weight, X_test):

        fitted_model = model.fit(self.X_train.values, self.y_train.values)

        return weight * model.predict(X_test.values)



    def combined_predict(self, X_test):

        prediction = np.zeros(X_test.shape[0])

        for model, weight in zip(self.models_to_combine_list, self.models_weights_list):

            prediction += self.fit_and_predict_weighted_model(model, weight, X_test)

        

        return np.exp(prediction)
Combined_predictor = CombinedModelsPredictor(processed_train_df, target)

final_predictions = Combined_predictor.combined_predict(processed_test_df)
submission = pd.DataFrame([{

    'Id': id,

    'SalePrice': sale_price

} for id, sale_price in zip(test_df.Id.values, final_predictions)])

submission.to_csv('submission.csv', index=False)