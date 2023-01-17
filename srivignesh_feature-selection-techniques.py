import numpy as np 

import pandas as pd 

from sklearn.feature_selection import SelectKBest, SelectFromModel, f_regression, chi2

from sklearn.feature_selection import RFE, RFECV

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor

import xgboost



pd.set_option('max.rows',500)

pd.set_option('max.columns',80)
'''Segregate the numeric and categoric columns'''

numeric_cols = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',

       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

       'MiscVal', 'MoSold', 'YrSold', 'SalePrice']



categoric_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']

preprocessed_train = pd.read_csv('../input/preprocessed-train-data/preprocessed_train_data.csv')

x_train, y_train = preprocessed_train[preprocessed_train.columns[:-1]], preprocessed_train[preprocessed_train.columns[-1]]

preprocessed_train.head()
'''Segregate numerical and categorical features'''

train_numeric = preprocessed_train[numeric_cols[:-1]]

x_train_numeric, y_train = train_numeric, preprocessed_train[numeric_cols[-1]]

train_categoric = preprocessed_train[categoric_cols]

x_train_categoric, y_train = train_categoric, preprocessed_train[numeric_cols[-1]]
'''Use f_regression for numeric columns'''

skb_numeric = SelectKBest(score_func = f_regression, k= 30)

skb_numeric.fit(x_train_numeric, y_train)

'''Get Support (Boolean array) for the columns from the instance'''

columns_selected_skb_numeric = x_train_numeric.columns[skb_numeric.get_support()]

x_train_skb_numeric = pd.DataFrame(skb_numeric.transform(x_train_numeric), columns = columns_selected_skb_numeric, index = x_train_numeric.index )

x_train_skb_numeric.head()
'''Chi2 test doesn't support negative values so square the dataset. Negative values are present in the dataset due to Z-Score normalization'''

x_train_categoric_sqr = x_train_categoric ** 2

'''Use chi2 for categoric columns'''

skb_categoric = SelectKBest(score_func = chi2, k= 30)

skb_categoric.fit(x_train_categoric_sqr, y_train)

'''Get Support (Boolean array) for the columns from the instance'''

columns_selected_skb_categoric = x_train_categoric_sqr.columns[skb_categoric.get_support()]

x_train_skb_categoric = pd.DataFrame(skb_categoric.transform(x_train_categoric), columns = columns_selected_skb_categoric, index =x_train_categoric.index )

x_train_skb_categoric.head()
'''Concatenate the selected features of numeric and categoric columns'''

x_train_skb = pd.concat([x_train_skb_numeric ,x_train_skb_categoric], axis =1)

x_train_skb.head()
xgb_model = xgboost.XGBRegressor(objective="reg:squarederror", random_state=42)

rfe = RFE(estimator= xgb_model, n_features_to_select = 25)

rfe.fit(x_train,y_train)

'''Select the columns that are already selected by RFE'''

columns_selected_rfe = x_train.columns[rfe.support_]

x_train_rfe = pd.DataFrame(rfe.transform(x_train), columns = columns_selected_rfe, index = x_train.index)

x_train_rfe.head()
rfecv = RFECV(estimator=xgb_model)

rfecv.fit(x_train, y_train)

'''Select the columns that are already selected by RFECV'''

columns_selected_rfecv = x_train.columns[rfecv.support_]

x_train_rfecv = pd.DataFrame(rfecv.transform(x_train), columns = columns_selected_rfecv, index = x_train.index)

x_train_rfecv.head()
lasso = Lasso(alpha = 0.3)

'''fit a LASSO model'''

lasso.fit(x_train, y_train)

'''To only select based on max_features, set threshold=-np.inf. Set prefit = True if the model is already fitted to the dataset.'''

sfm_lasso = SelectFromModel(estimator=lasso, prefit= True, max_features=65, threshold=-np.inf)

lasso_selected_columns = x_train.columns[sfm_lasso.get_support()]

x_train_lasso = pd.DataFrame(sfm_lasso.transform(x_train), columns = lasso_selected_columns, index = x_train.index)

x_train_lasso.head()
'''Any regressor can be used. Here Decision Tree is used'''

dec_tree_model = DecisionTreeRegressor()

dec_tree_model.fit(x_train, y_train)



'''To only select based on max_features, set threshold=-np.inf. Set prefit = True if the model is already fitted to the dataset.'''

sfm = SelectFromModel(estimator=dec_tree_model, prefit= True, max_features=65, threshold=-np.inf)



'''Selected columns'''

columns_selected_sfm = preprocessed_train.columns[:-1][sfm.get_support()]

x_train_sfm = pd.DataFrame(sfm.transform(x_train), columns = columns_selected_sfm, index = x_train.index)

x_train_sfm.head()