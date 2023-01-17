import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from scipy import stats
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
train_raw = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')
ntrain = train_raw.shape[0]
ntest = test_raw.shape[0]

y_train = train_raw.SalePrice.values

# concatenate training and test data into all_data
all_data = pd.concat((train_raw, test_raw)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data_na = all_data.isnull().sum()
all_data_dummies = pd.get_dummies(all_data)
all_data_dummies["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
all_data_dummies["GarageArea"] = all_data.groupby("Neighborhood")["GarageArea"].transform(
    lambda x: x.fillna(x.median()))
#all_data_dummies['Neighborhood3'] = all_data.Neighborhood
#all_data_dummies.columns.values
train2 = all_data_dummies[:ntrain]
test2 = all_data_dummies[ntrain:]
#avg_sale_price = train_raw.groupby('Neighborhood').mean().SalePrice
all_data_dummies.columns.values
cols_to_keep = ['GrLivArea', 'OverallQual' , 'FullBath', 'OverallCond', 'GarageArea', 
                '2ndFlrSF', '1stFlrSF' ,  'TotRmsAbvGrd', 'LotFrontage', 'LotArea',
                'YrSold', 'KitchenQual_Ex', 'KitchenQual_Fa', 'KitchenQual_Gd', 
                'LandContour_Bnk', 'LandContour_HLS',
                'CentralAir_N',
       'CentralAir_Y',
       'LandContour_Low', 'LandContour_Lvl', 'LandSlope_Gtl',
       'LandSlope_Mod', 'LandSlope_Sev',
                'Neighborhood_BrDale', 'Neighborhood_BrkSide',
               'Neighborhood_ClearCr', 'Neighborhood_CollgCr',
               'Neighborhood_Crawfor', 'Neighborhood_Edwards',
               'Neighborhood_Gilbert', 'Neighborhood_IDOTRR',
               'Neighborhood_MeadowV', 'Neighborhood_Mitchel',
               'Neighborhood_NAmes', 'Neighborhood_NPkVill',
               'Neighborhood_NWAmes', 'Neighborhood_NoRidge',
               'Neighborhood_NridgHt', 'Neighborhood_OldTown',
               'Neighborhood_SWISU', 'Neighborhood_Sawyer',
               'Neighborhood_SawyerW', 'Neighborhood_Somerst',
               'Neighborhood_StoneBr', 'Neighborhood_Timber',
               'Neighborhood_Veenker', 'BldgType_1Fam', 'BldgType_2fmCon', 'BldgType_Duplex',
               'BldgType_Twnhs', 'BldgType_TwnhsE',
               'SaleType_New', 'SaleType_WD',
                'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_1Story',
       'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story',
       'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'LotArea',
               'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort',
       'GarageType_Detchd',
               'ExterCond_Fa', 'ExterCond_Gd', 'ExterCond_Po', 'ExterCond_TA',
       'ExterQual_Ex', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'SaleCondition_Abnorml',
       'SaleCondition_AdjLand', 'SaleCondition_Alloca',
       'SaleCondition_Family', 'SaleCondition_Normal',
       'SaleCondition_Partial']
cols_to_keep = ['GrLivArea', 'OverallQual' , 'FullBath', 'OverallCond', '1stFlrSF' ,  'TotRmsAbvGrd', 'LotFrontage', 'LotArea']
train_keep_cols = train2[cols_to_keep]
test_keep_cols = test2[cols_to_keep]
from sklearn import linear_model
train_for_model = train_keep_cols
test_for_model = test_keep_cols
#model = linear_model.Ridge(alpha = .0001)
#model = linear_model.ElasticNet(alpha = 1)

from sklearn.metrics import mean_squared_error
from sklearn import ensemble
model = ensemble.GradientBoostingRegressor(n_estimators = 10000, max_depth = 3, learning_rate = .01, subsample = .6)
#from sklearn import model_selection
#model_selection.cross_val_score(model, train_for_model, y_train, cv = 6) #scoring = 'neg_mean_squared_error')|
model.fit(train_for_model, y_train)
predictions = model.predict(test_for_model)
test_df = train_keep_cols.copy()
predictions_train = model.predict(train_for_model)
test_df['predictions'] = predictions_train
test_df['salePrices'] = train_raw['SalePrice']
test_df.head(20)
#from sklearn.linear_model import LinearRegression
#lm = LinearRegression()
#lm.fit(xgb_train, y_train)
#predictions = lm.predict(xgb_test)
test_raw['Id'].shape
submissionDF = pd.DataFrame()
submissionDF['Id'] = test_raw['Id']
submissionDF['SalePrice'] = predictions
submissionDF.head()
submissionDF.to_csv('test_4.csv', index = False)
