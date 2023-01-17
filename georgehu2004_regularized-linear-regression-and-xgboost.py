import pandas as pd

import numpy as np

from scipy.stats import skew

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

import xgboost as xgb

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.metrics import mean_absolute_error, mean_squared_log_error

from sklearn.feature_selection import VarianceThreshold



sns.set_style('darkgrid')

plt.rcParams['figure.figsize'] = 7, 5
# Read in training data

house_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')



y = house_data['SalePrice']

X = house_data.drop('SalePrice', axis=1)
# Read in test data

X_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
# Dimensions of training data (1460 rows, 79 columns)

display(X.shape)

# Examine first five rows of training data

X.head()
# Data types of variables (43 categorical, 36 numerical)

X.dtypes.value_counts()
# Summary statistics for numerical variables

num_cols = [col for col in X.columns if X.dtypes[col] != 'object']

X[num_cols].describe()
# Summary statistics for categorical variables

cat_cols = [col for col in X.columns if X.dtypes[col] == 'object']

X[cat_cols].describe()
# Summary statistics for SalePrice (target variable)

y.describe()
# Histogram of SalePrice (target variable)

y_thousands = y/1000

sns.distplot(y_thousands)

plt.title('Histogram of SalePrice (Target Variable)')

plt.xlabel('Sale Price (Thousands USD)');
# Correlation matrix (based on https://seaborn.pydata.org/examples/many_pairwise_correlations.html)

plt.figure(figsize=(12, 8))

corr_matrix = house_data.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))

cmap = sns.diverging_palette(220, 10, as_cmap=True)



with sns.axes_style('white'):

    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('House Prices Correlation Matrix');
# Ten features most correlated with SalePrice (according to Pearson correlation)

top_ten_corr_features = corr_matrix.nlargest(n=11, columns='SalePrice')['SalePrice'][1:]

top_ten_corr_features
# Scatterplots of SalePrice vs. top ten correlated features (regression line in red)

sns.pairplot(data=house_data, y_vars=['SalePrice'], x_vars=top_ten_corr_features.index[:5], height=4, aspect=0.8, kind='reg', plot_kws={'line_kws':{'color':'red'}})

sns.pairplot(data=house_data, y_vars=['SalePrice'], x_vars=top_ten_corr_features.index[5:], height=4, aspect=0.8, kind='reg', plot_kws={'line_kws':{'color':'red'}});
# Number of houses and garages built by year

sns.lineplot(data=house_data.YearBuilt.value_counts(), marker='o', label='Houses')

sns.lineplot(data=house_data.GarageYrBlt.value_counts(), marker='o', label='Garages')

plt.title('Number of Houses and Garages Built by Year')

plt.ylabel('Number Built')

plt.xlabel('Year')

plt.legend();
# Number of Houses Sold by Year-Month

YearMonthSold = pd.to_datetime(house_data.YrSold.apply(str) + '-' + house_data.MoSold.apply(str) + '-1')

sns.lineplot(data=YearMonthSold.value_counts(), marker='o', color='g')

plt.title('Number of Houses Sold by Year-Month')

plt.ylabel('Number of Houses Sold')

plt.xticks(rotation='vertical');
sns.lineplot(data=house_data.groupby(YearMonthSold)['SalePrice'].median()/1000, marker='o', color='navy')

plt.title('Median SalePrice by Year-Month')

plt.ylabel('Sale Price (Thousands USD)')

plt.xticks(rotation='vertical');
def impute(df, column, value):

    """

    Function that performs missing value imputation on a column or columns in a dataframe.

    

    Inputs:

    df (dataframe) -- dataframe of the column(s)

    column (string or list of strings) -- column name(s) of the column(s) to impute

    value (int, float, or string) -- value to impute

    """

    df[column] = df[column].fillna(value)
# Number of missing values in each column of the training set (excluding columns with no missing values)

X.isna().sum()[X.isna().sum() > 0]
# Number of missing values in each column of the test set (excluding columns with no missing values)

X_test.isna().sum()[X_test.isna().sum() > 0]
# Differences in variables with missing values between training and test sets

print('Variable with NA in training but not test set: ', set(X.columns[X.isna().any()].tolist()) - set(X_test.columns[X_test.isna().any()].tolist()))

print('Variables with NA in test but not training set: ', set(X_test.columns[X_test.isna().any()].tolist()) - set(X.columns[X.isna().any()].tolist()))
impute(X, 'MSZoning', X['MSZoning'].value_counts().idxmax())

impute(X_test, 'MSZoning', X['MSZoning'].value_counts().idxmax())
X['LotFrontage'] = X['LotFrontage'].fillna(X.groupby('Neighborhood')['LotFrontage'].transform('median'))



# Create column for LotFrontage median grouped by Neighborhood

lotfrontage_median = pd.DataFrame(X.groupby('Neighborhood')['LotFrontage'].median()).rename(columns={'LotFrontage':'LotFrontage_median'})



# Merge LotFrontage median with test set

X_test_tmp = X_test.reset_index().merge(lotfrontage_median, on='Neighborhood').set_index('Id')



# Fill missing LotFrontage in test set with LotFrontage median

X_test_tmp['LotFrontage'] = X_test_tmp['LotFrontage'].fillna(X_test_tmp['LotFrontage_median'])



# Drop LotFrontage median

X_test = X_test_tmp.drop('LotFrontage_median', axis=1)
impute(X, 'Alley', 'None')

impute(X_test, 'Alley', 'None')
impute(X, 'Utilities', X['Utilities'].value_counts().idxmax())

impute(X_test, 'Utilities', X['Utilities'].value_counts().idxmax())
impute(X, 'Exterior1st', X['Exterior1st'].value_counts().idxmax())

impute(X_test, 'Exterior1st', X['Exterior1st'].value_counts().idxmax())



impute(X, 'Exterior2nd', X['Exterior2nd'].value_counts().idxmax())

impute(X_test, 'Exterior2nd', X['Exterior2nd'].value_counts().idxmax())
impute(X, 'MasVnrType', 'None')

impute(X_test, 'MasVnrType', 'None')



impute(X, 'MasVnrArea', 0)

impute(X_test, 'MasVnrArea', 0)
# Categorical basement-related columns

Bsmt_cat_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

# Numerical basement-realted columns

Bsmt_num_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']



impute(X, Bsmt_cat_cols, 'None')

impute(X_test, Bsmt_cat_cols, 'None')



impute(X, Bsmt_num_cols, 0)

impute(X_test, Bsmt_num_cols, 0)
impute(X, 'KitchenQual', X['KitchenQual'].value_counts().idxmax())

impute(X_test, 'KitchenQual', X['KitchenQual'].value_counts().idxmax())
impute(X, 'Functional', X['Functional'].value_counts().idxmax())

impute(X_test, 'Functional', X['Functional'].value_counts().idxmax())
impute(X, 'FireplaceQu', 'None')

impute(X_test, 'FireplaceQu', 'None')
# Categorical garage-related columns

Garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

# Numerical garage-realted columns

Garage_num_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars']



impute(X, Garage_cat_cols, 'None')

impute(X_test, Garage_cat_cols, 'None')



impute(X, Garage_num_cols, 0)

impute(X_test, Garage_num_cols, 0)
impute(X, 'PoolQC', 'None')

impute(X_test, 'PoolQC', 'None')
impute(X, 'Fence', 'None')

impute(X_test, 'Fence', 'None')
impute(X, 'MiscFeature', 'None')

impute(X_test, 'MiscFeature', 'None')
impute(X, 'SaleType', X['SaleType'].value_counts().idxmax())

impute(X_test, 'SaleType', X['SaleType'].value_counts().idxmax())
impute(X, 'Electrical', X['Electrical'].value_counts().idxmax())

impute(X_test, 'Electrical', X['Electrical'].value_counts().idxmax())
# Make sure that there are no more NA

print('Number of NA in training set after imputation: ', X.isna().sum().sum())

print('Number of NA in test set after imputation: ', X_test.isna().sum().sum())
def create_features(X):

    # Total house square footage

    X['TotalSF'] = X['TotalBsmtSF'] + X['GrLivArea']



    # Total porch square footage

    X['TotalPorchSF'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['3SsnPorch'] + X['ScreenPorch']



    # Total bathrooms

    X['TotalBath'] = X['BsmtFullBath'] + 0.5 * X['BsmtHalfBath'] + X['FullBath'] + 0.5 * X['HalfBath']



    # Total rooms above grade (including bathrooms)

    X['TotalRooms'] = X['TotRmsAbvGrd'] + X['FullBath'] + 0.5 * X['HalfBath']

    

    # Does the house have a garage?

    X['HasGarage'] = X['GarageArea'] > 0



    # Does the house have a pool?

    X['HasPool'] = X['PoolArea'] > 0



    # Does the house have a fireplace?

    X['HasFireplace'] = X['Fireplaces'] > 0



    # Does the house have a basement?

    X['HasBasement'] = X['TotalBsmtSF'] > 0



    # Does the house have a 2nd floor?

    X['Has2ndFloor'] = X['2ndFlrSF'] > 0

    

    # Has the house been remodeled?

    X['IsRemodeled'] = X['YearBuilt'] != X['YearRemodAdd']

    

    return X



X = create_features(X)

X_test = create_features(X_test)
# # Create a temporary dataframe with features and SalePrice

# df_temp = X.copy()

# df_temp['SalePrice'] = y



# # Check ten features most correlated with SalePrice

# corr_matrix = df_temp.corr()

# top_ten_corr_features = corr_matrix.nlargest(n=11, columns='SalePrice')['SalePrice'][1:]

# top_ten_corr_features
# # Create polynomial features for the ten features most correlated with SalePrice

# for col in top_ten_corr_features.index:

#     X[col + '^2'] = X[col] ** 2

#     X_test[col + '^2'] = X_test[col] ** 2

    

#     X[col + '^3'] = X[col] ** 3

#     X_test[col + '^3'] = X_test[col] ** 3

    

#     X[col + '^0.5'] = X[col] ** 0.5

#     X_test[col + '^0.5'] = X_test[col] ** 0.5
# Convert these numerical variables to categorical

num_cols_to_cat = ['MSSubClass', 'MoSold', 'YrSold']

X[num_cols_to_cat] = X[num_cols_to_cat].astype(str)

X_test[num_cols_to_cat] = X_test[num_cols_to_cat].astype(str)
# Lists of categorical and numerical variables

cat_cols = [col for col in X.columns if X.dtypes[col] == 'object']

num_cols = [col for col in X.columns if X.dtypes[col] in ['int64', 'float64']]
# Ordinal encode the following categorical variables

def ordinal_encoder(X):

    X = X.replace({

        'Street': {'Pave':1, 'Grvl':2},

        'Alley': {'None':0, 'Pave':1, 'Grvl':2},

        'LotShape': {'IR3':1, 'IR2':2, 'IR1':3, 'Reg':4},

        'Utilities': {'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4},

        'LandSlope': {'Sev':1, 'Mod':2, 'Gtl':3},

        'ExterQual': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},

        'ExterCond': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},

        'BsmtQual': {'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},

        'BsmtCond': {'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},

        'BsmtExposure': {'None':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4},

        'BsmtFinType1': {'None':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6},

        'BsmtFinType2': {'None':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6},

        'HeatingQC': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},

        'KitchenQual': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},

        'Functional': {'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8},

        'FireplaceQu': {'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},

        'GarageFinish': {'None':0, 'Unf':1, 'RFn':2, 'Fin':3},

        'GarageQual': {'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},

        'GarageCond': {'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},

        'PavedDrive': {'N':0, 'P':1, 'Y':2},

        'PoolQC': {'None':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4},

        'Fence': {'None':0, 'MnWw':1, 'GdWo':2, 'MnPrv':3, 'GdPrv':4},

        'Electrical':{'Mix':1, 'FuseP':2, 'FuseF':3, 'FuseA':4, 'SBrkr':5}

    })

    return X



X = ordinal_encoder(X)

X_test = ordinal_encoder(X_test)
# Apply one-hot encoding to training and test sets

X = pd.get_dummies(X)

X_test = pd.get_dummies(X_test)
# Training set now has more columns test set due to dummy variables only found in training set

print(X.shape)

print(X_test.shape)
# Find columns that are in training set but not test set

missing_cols = set(X.columns) - set(X_test.columns)



# Add these columns to test set filled with 0's

for col in missing_cols:

    X_test[col] = 0

    

# Align columns of training and test sets

X_test = X_test[X.columns]



print(X.shape)

print(X_test.shape)
# Histogram of SalePrice (target variable)

y_thousands = y/1000

sns.distplot(y_thousands)

plt.title('Histogram of SalePrice (Target Variable)')

plt.xlabel('Sale Price (Thousands USD)');
# Log transform SalePrice: log(1 + SalePrice)

y = np.log1p(y)



# Histogram of SalePrice (log transformed)

sns.distplot(y)

plt.title('Histogram of SalePrice (Log Transformed)')

plt.xlabel('Sale Price (Log Transformed)');
# Calculate skewness for numerical variables

skewness = X[num_cols].apply(lambda x: skew(x))

skewed_cols = skewness[abs(skewness) > 0.5].index



# Log transform skewed numerical variables 

X[skewed_cols] = np.log1p(X[skewed_cols])

X_test[skewed_cols] = np.log1p(X_test[skewed_cols])
# Set variance threshold to 0.01

sel = VarianceThreshold(threshold=0.01)

sel.fit(X)

print('Number of features remaining:', sel.get_support().sum())
# Remove any feature with variance less than 0.01

X = X[[col for col in X.columns[sel.get_support()]]]

X_test = X_test[[col for col in X_test.columns[sel.get_support()]]]
# Feature scaling

scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)

X_test_scaled = scaler.transform(X_test)



# Scaling removes column names and index; get them back

X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
# Fuction to calculate RMSE

def CV_rmse(estimator, X, y, cv):

    rmse = -cross_val_score(estimator=estimator, X=X, y=y, cv=cv, scoring='neg_root_mean_squared_error')

    return rmse
linear_model = LinearRegression()



rmse = CV_rmse(estimator=linear_model, X=X_scaled, y=y, cv=10)

print('Linear Regression RMSE:', rmse.mean())
# # Tuning

# params = {'alpha': range(1, 50, 2)}



# ridge_cv = GridSearchCV(estimator=Ridge(max_iter=20000, random_state=1), param_grid=params, scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)

# ridge_cv.fit(X_scaled, y)

# print('Ridge Regression Best Alpha:', ridge_cv.best_params_['alpha'])

# print('Ridge Regression RMSE:', -ridge_cv.best_score_)
ridge_cv = Ridge(alpha=25, max_iter=20000, random_state=1)

ridge_cv.fit(X_scaled, y)



rmse = CV_rmse(estimator=ridge_cv, X=X_scaled, y=y, cv=10)

print('Ridge Regression RMSE:', rmse.mean())
# Ridge test predictions 

ridge_pred = np.expm1(ridge_cv.predict(X_test_scaled))
# # Tuning

# params = {'alpha': [0.0005, 0.0006, 0.0007, 0.0008, 0.0009]}



# lasso_cv = GridSearchCV(estimator=Lasso(max_iter=20000, random_state=1), param_grid=params, scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)

# lasso_cv.fit(X_scaled, y)

# print('Lasso Regression Best Alpha:', lasso_cv.best_params_['alpha'])

# print('Lasso Regression RMSE:', -lasso_cv.best_score_)
lasso_cv = Lasso(alpha=0.0008, max_iter=20000, random_state=1)

lasso_cv.fit(X_scaled, y)



rmse = CV_rmse(estimator=lasso_cv, X=X_scaled, y=y, cv=10)

print('Lasso Regression RMSE:', rmse.mean())
# Lasso test predictions 

lasso_pred = np.expm1(lasso_cv.predict(X_test_scaled))
# # Tuning

# params = {'alpha': [0.0005, 0.0006, 0.0007, 0.0008, 0.0009],

#          'l1_ratio': np.linspace(0.7, 1, 10)}



# elasticnet_cv = GridSearchCV(estimator=ElasticNet(max_iter=20000, random_state=1), param_grid=params, scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)

# elasticnet_cv.fit(X_scaled, y)

# print('ElasticNet Regression Best Alpha:', elasticnet_cv.best_params_['alpha'])

# print('ElasticNet Regression Best L1 Ratio:', elasticnet_cv.best_params_['l1_ratio'])

# print('ElasticNet Regression RMSE:', -elasticnet_cv.best_score_)
elasticnet_cv = ElasticNet(alpha=0.0008, l1_ratio=1.0, max_iter=20000, random_state=1)

elasticnet_cv.fit(X_scaled, y)



rmse = CV_rmse(estimator=elasticnet_cv, X=X_scaled, y=y, cv=10)

print('ElasticNet Regression RMSE:', rmse.mean())
# ElasticNet test predictions 

elasticnet_pred = np.expm1(elasticnet_cv.predict(X_test_scaled))
xgb_model = xgb.XGBRegressor(random_state=1)



rmse = CV_rmse(estimator=xgb_model, X=X, y=y, cv=10)

print('XGBoost (untuned) RMSE:', rmse.mean())
# XGBoost with tuned parameters

xbg_model_tuned = xgb.XGBRegressor(max_depth=3, min_child_weight=6, learning_rate=0.03, subsample=0.7, colsample_bytree=0.8, n_estimators=727, random_state=1)



xbg_model_tuned.fit(X, y)

rmse = CV_rmse(estimator=xbg_model_tuned, X=X, y=y, cv=10)

print('XGBoost (tuned) RMSE:', rmse.mean())
# XGBoost test predictions 

xgb_pred = np.expm1(xbg_model_tuned.predict(X_test))
avg_pred = 0.2 * ridge_pred + 0.4 * lasso_pred + 0.4 * xgb_pred
#avg_pred = np.array([ridge_pred, lasso_pred, xgb_pred]).mean(axis=0)
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                        'SalePrice': avg_pred})

output.to_csv('submission.csv', index=False)
# from IPython.display import FileLink

# FileLink('submission.csv')