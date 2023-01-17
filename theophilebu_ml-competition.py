import pandas as pd

import numpy as np

import copy



from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns





from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax
# Path of the file to read. We changed the directory structure to simplify submitting to a competition

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.drop(['Id'], axis=1, inplace=True)

testId = test.Id

test.drop(['Id'], axis=1, inplace=True)
train = train[train.GrLivArea < 4500]

#train.reset_index(drop=True, inplace=True)

#train["SalePrice"] = np.log1p(train["SalePrice"])

y = train['SalePrice']#.reset_index(drop=True)



X = train.drop(['SalePrice'], axis=1)



#Concat dataset for cleaning

train_objs_num = len(X)

features = pd.concat(objs=[X, test], axis=0)
#code from https://www.kaggle.com/niteshx2/top-50-beginners-stacking-lgb-xgb



# Since these column are actually a category , using a numerical number will lead the model to assume

# that it is numerical , so we convert to string .

features['MSSubClass'] = features['MSSubClass'].apply(str)

features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)



## Filling these columns With most suitable value for these columns 

features['Functional'] = features['Functional'].fillna('Typ') 

features['Electrical'] = features['Electrical'].fillna("SBrkr") 

features['KitchenQual'] = features['KitchenQual'].fillna("TA") 

features["PoolQC"] = features["PoolQC"].fillna("None")



## Filling these with MODE , i.e. , the most frequent value in these columns .

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) 

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])



### Missing data in GarageYrBit most probably means missing Garage , so replace NaN with zero . 



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)



for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    features[col] = features[col].fillna('None')

    

### Same with basement

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')



features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



objects = []

for i in features.columns:

    if features[i].dtype == object:

        objects.append(i)

features.update(features[objects].fillna('None'))

print(objects)

 

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics.append(i)

features.update(features[numerics].fillna(0))

numerics[1:10]



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics2.append(i)

skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



for i in skew_index:

    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))

    

# Removing features that are not very useful . This can be understood only by doing proper EDA on data

features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)



# Adding new features . Make sure that you understand this. 



features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']



features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])



features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))



features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])

## For ex, if PoolArea = 0 , Then HasPool = 0 too



features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)





final_features = pd.get_dummies(features).reset_index(drop=True)



for column in final_features.columns:

    final_features[column].fillna((final_features[column].mean()), inplace=True)

    

final_features.shape
for col in final_features.columns:

    if final_features[col].isna().any():

        print(col)
data = copy.copy(final_features[:train_objs_num])

test = copy.copy(final_features[train_objs_num:])

print(data.shape)

print(test.shape)
"""

import xgboost as xgb

import xgbfir



#doing all the XGBoost magic

xgb_rmodel = xgb.XGBRegressor().fit(data, y)



#saving to file with proper feature names

xgbfir.saveXgbFI(xgb_rmodel, feature_names=dataset.columns.values, OutputXlsxFile='housepricing.xlsx')



xl = pd.ExcelFile("housepricing.xlsx")

depth1 = xl.parse("Interaction Depth 0")

depth2 = xl.parse("Interaction Depth 1")

depth3 = xl.parse("Interaction Depth 2")

depth1 =depth1.Interaction.values.tolist()

depth1 = [(x.split('|')[0]) for x in depth1]

depth2 =depth2.Interaction.values.tolist()

depth2 = [(x.split('|')[0],x.split('|')[1]) for x in depth2]

depth3 =depth3.Interaction.values.tolist()

depth3 = [(x.split('|')[0],x.split('|')[1],x.split('|')[2]) for x in depth3]

"""
depth1 = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'YearRemodAdd', 'YearBuilt', 'GarageCars', 'BsmtFinSF1', 'LotArea', 'Fireplaces', 'OverallCond', 'GarageArea', 'BsmtQual_Ex', 'GarageYrBlt', 'GarageType_Attchd', 'CentralAir_N', '1stFlrSF', 'LotFrontage', 'MSZoning_C (all)', 'BsmtExposure_Gd', 'KitchenQual_Ex', 'Neighborhood_Crawfor', 'SaleCondition_Abnorml', 'Functional_Typ', 'MSZoning_RM', 'MSSubClass_30', 'BsmtFinType1_GLQ', 'WoodDeckSF', 'OpenPorchSF', '2ndFlrSF', 'SaleType_New', 'KitchenAbvGr', 'KitchenQual_TA', 'MSZoning_RL', 'ScreenPorch', 'GarageQual_TA', 'BsmtFullBath', 'MoSold_2', 'Exterior1st_BrkFace', 'HalfBath', 'BsmtUnfSF', 'SaleCondition_Family', 'TotRmsAbvGrd', 'PoolArea', 'BsmtExposure_No', 'Condition1_Norm', 'Condition1_Artery', 'LotShape_Reg', 'HeatingQC_Ex', 'Exterior1st_AsbShng', 'Neighborhood_OldTown', 'Alley_Grvl', 'MSSubClass_50', 'Heating_Grav', 'EnclosedPorch', 'Functional_Min2', 'RoofStyle_Flat', 'Neighborhood_Sawyer', 'MoSold_11', 'BsmtQual_Gd', 'Electrical_SBrkr', 'PavedDrive_Y', 'BsmtCond_TA', 'Neighborhood_StoneBr', 'Condition1_PosA', 'MSSubClass_90', 'BsmtFinType2_LwQ', 'Exterior1st_BrkComm', 'Neighborhood_BrkSide', 'Exterior1st_VinylSd', 'MoSold_8', 'MiscVal', 'YrSold_2009', 'BsmtFinSF2', 'LandContour_Lvl', 'Neighborhood_ClearCr', 'HouseStyle_SLvl', 'MSZoning_FV', 'FullBath', 'BedroomAbvGr', 'Neighborhood_Mitchel', 'BsmtFinType1_ALQ', 'YrSold_2010', 'Neighborhood_Edwards', 'LotConfig_Inside', 'Functional_Maj2', 'Condition1_PosN', 'LotConfig_CulDSac', 'FireplaceQu_TA', 'Exterior1st_HdBoard', 'PavedDrive_N', 'MasVnrType_BrkFace', 'BsmtHalfBath', 'YrSold_2006', 'LotConfig_FR2', 'RoofStyle_Gable', 'MoSold_3', 'Fence_MnPrv', 'HeatingQC_Gd']  

depth2 = [('OverallQual', 'TotalBsmtSF'), ('GrLivArea', 'TotalBsmtSF'), ('BsmtFinSF1', 'GrLivArea'), ('GarageArea', 'OverallQual'), ('BsmtFinSF1', 'YearRemodAdd'), ('GrLivArea', 'YearBuilt'), ('GarageYrBlt', 'TotalBsmtSF'), ('GarageCars', 'OverallQual'), ('GrLivArea', 'YearRemodAdd'), ('GarageType_Attchd', 'GrLivArea'), ('CentralAir_N', 'GrLivArea'), ('Fireplaces', 'YearBuilt'), ('Fireplaces', 'GarageCars'), ('CentralAir_N', 'TotalBsmtSF'), ('BsmtQual_Ex', 'GrLivArea'), ('Fireplaces', 'YearRemodAdd'), ('GrLivArea', 'GrLivArea'), ('TotalBsmtSF', 'YearRemodAdd'), ('GrLivArea', 'LotArea'), ('GarageYrBlt', 'LotFrontage'), ('LotArea', 'YearRemodAdd'), ('OverallCond', 'YearBuilt'), ('GrLivArea', 'OverallCond'), ('LotArea', 'OverallCond'), ('1stFlrSF', 'BsmtFinSF1'), ('CentralAir_N', 'GarageArea'), ('GarageType_Attchd', 'OpenPorchSF'), ('BsmtFinSF1', 'OverallCond'), ('OverallCond', 'YearRemodAdd'), ('CentralAir_N', 'MSZoning_C (all)'), ('LotFrontage', 'YearBuilt'), ('GrLivArea', 'MSZoning_RM'), ('BsmtFinSF1', 'GarageArea'), ('1stFlrSF', 'GarageArea'), ('BsmtFinType1_GLQ', 'LotArea'), ('OverallCond', 'TotalBsmtSF'), ('GrLivArea', 'MSSubClass_30'), ('Fireplaces', 'GrLivArea'), ('LotArea', 'OverallQual'), ('Functional_Min2', 'GrLivArea'), ('Fireplaces', 'GarageQual_TA'), ('BsmtQual_Ex', 'OverallCond'), ('GarageYrBlt', 'OpenPorchSF'), ('BsmtFinSF1', 'LotArea'), ('1stFlrSF', 'YearRemodAdd'), ('GarageYrBlt', 'OverallQual'), ('GrLivArea', 'KitchenAbvGr'), ('MSSubClass_30', 'YearRemodAdd'), ('2ndFlrSF', 'CentralAir_N'), ('MSZoning_RL', 'YearRemodAdd'), ('GrLivArea', 'KitchenQual_Ex'), ('BsmtUnfSF', 'TotalBsmtSF'), ('CentralAir_N', 'MSZoning_RM'), ('BsmtFinSF1', 'KitchenQual_Ex'), ('BsmtFinType1_GLQ', 'TotalBsmtSF'), ('Exterior1st_BrkComm', 'OverallQual'), ('GarageYrBlt', 'GrLivArea'), ('1stFlrSF', 'GrLivArea'), ('GrLivArea', 'OverallQual'), ('GarageArea', 'GrLivArea'), ('KitchenQual_TA', 'OverallQual'), ('BsmtExposure_Gd', 'YearRemodAdd'), ('GarageYrBlt', 'OverallCond'), ('Exterior1st_BrkComm', 'GrLivArea'), ('1stFlrSF', 'TotRmsAbvGrd'), ('GarageQual_TA', 'MSZoning_C (all)'), ('1stFlrSF', 'MSSubClass_30'), ('GrLivArea', 'RoofStyle_Flat'), ('GarageArea', 'LotArea'), ('LotArea', 'MSZoning_C (all)'), ('SaleCondition_Family', 'WoodDeckSF'), ('BsmtFullBath', 'OverallCond'), ('BsmtFinSF1', 'TotalBsmtSF'), ('MoSold_2', 'SaleType_New'), ('BsmtFinType1_GLQ', 'GrLivArea'), ('KitchenQual_Ex', 'WoodDeckSF'), ('1stFlrSF', 'BsmtExposure_Gd'), ('MSZoning_RL', 'OverallCond'), ('GarageQual_TA', 'OverallQual'), ('Heating_Grav', 'MSZoning_RL'), ('MSZoning_C (all)', 'YearBuilt'), ('GarageArea', 'YearBuilt'), ('BsmtExposure_Gd', 'LotArea'), ('GarageCars', 'GrLivArea'), ('LotArea', 'Neighborhood_Crawfor'), ('2ndFlrSF', 'OverallQual'), ('Alley_Grvl', 'GrLivArea'), ('BsmtFinSF1', 'Neighborhood_Sawyer'), ('MoSold_2', 'Neighborhood_Crawfor'), ('MSZoning_C (all)', 'WoodDeckSF'), ('BsmtFinSF1', 'Neighborhood_Crawfor'), ('OverallCond', 'OverallQual'), ('GrLivArea', 'LotShape_Reg'), ('Neighborhood_OldTown', 'OverallCond'), ('Functional_Typ', 'GrLivArea'), ('GrLivArea', 'HalfBath'), ('BsmtExposure_Gd', 'LotFrontage'), ('Exterior1st_BrkFace', 'MSZoning_C (all)'), ('1stFlrSF', 'KitchenQual_TA'), ('LotFrontage', 'MoSold_2')]

depth3 = [('GarageYrBlt', 'LotFrontage', 'TotalBsmtSF'), ('BsmtFinSF1', 'GrLivArea', 'YearRemodAdd'), ('1stFlrSF', 'BsmtFinSF1', 'YearRemodAdd'), ('GarageType_Attchd', 'GrLivArea', 'OpenPorchSF'), ('1stFlrSF', 'GarageArea', 'OverallQual'), ('GrLivArea', 'OverallCond', 'YearBuilt'), ('GrLivArea', 'LotFrontage', 'YearBuilt'), ('CentralAir_N', 'MSZoning_C (all)', 'TotalBsmtSF'), ('2ndFlrSF', 'CentralAir_N', 'TotalBsmtSF'), ('Exterior1st_BrkComm', 'OverallQual', 'TotalBsmtSF'), ('BsmtQual_Ex', 'GrLivArea', 'TotalBsmtSF'), ('GrLivArea', 'MSSubClass_30', 'YearRemodAdd'), ('GarageYrBlt', 'OpenPorchSF', 'TotalBsmtSF'), ('BsmtQual_Ex', 'GrLivArea', 'MSZoning_RM'), ('GrLivArea', 'YearBuilt', 'YearRemodAdd'), ('BsmtUnfSF', 'TotalBsmtSF', 'YearRemodAdd'), ('OverallCond', 'TotalBsmtSF', 'YearRemodAdd'), ('Exterior1st_BrkComm', 'GrLivArea', 'TotalBsmtSF'), ('BsmtFinSF1', 'GrLivArea', 'OverallCond'), ('LotArea', 'OverallCond', 'YearRemodAdd'), ('GarageYrBlt', 'OverallQual', 'TotalBsmtSF'), ('BsmtFinSF1', 'GrLivArea', 'GrLivArea'), ('BsmtFinSF1', 'LotArea', 'OverallCond'), ('CentralAir_N', 'GarageArea', 'MSZoning_C (all)'), ('LotArea', 'MSSubClass_30', 'YearRemodAdd'), ('CentralAir_N', 'GarageArea', 'MSZoning_RM'), ('GrLivArea', 'GrLivArea', 'TotalBsmtSF'), ('GarageArea', 'KitchenQual_TA', 'OverallQual'), ('Alley_Grvl', 'GrLivArea', 'LotArea'), ('1stFlrSF', 'GrLivArea', 'YearRemodAdd'), ('BsmtFinType1_GLQ', 'LotArea', 'TotalBsmtSF'), ('GrLivArea', 'GrLivArea', 'LotArea'), ('Fireplaces', 'GrLivArea', 'TotalBsmtSF'), ('BsmtQual_Ex', 'LotArea', 'OverallCond'), ('BsmtFinSF1', 'GarageArea', 'GrLivArea'), ('Fireplaces', 'GarageQual_TA', 'MSZoning_C (all)'), ('BsmtFinType1_GLQ', 'GrLivArea', 'LotArea'), ('Fireplaces', 'GrLivArea', 'RoofStyle_Flat'), ('BedroomAbvGr', 'BsmtFinSF1', 'GrLivArea'), ('Functional_Min2', 'GrLivArea', 'YearBuilt'), ('LotArea', 'OverallCond', 'OverallQual'), ('BsmtQual_Ex', 'GrLivArea', 'OverallCond'), ('Fireplaces', 'GarageQual_TA', 'OverallQual'), ('1stFlrSF', 'TotRmsAbvGrd', 'YearRemodAdd'), ('GrLivArea', 'GrLivArea', 'KitchenAbvGr'), ('1stFlrSF', 'MSSubClass_30', 'YearRemodAdd'), ('MSZoning_RL', 'OverallCond', 'YearRemodAdd'), ('BsmtFinSF1', 'GrLivArea', 'OverallQual'), ('Heating_Grav', 'MSZoning_RL', 'YearRemodAdd'), ('BsmtFinSF1', 'GrLivArea', 'KitchenQual_Ex'), ('GrLivArea', 'MSZoning_C (all)', 'YearBuilt'), ('GarageArea', 'GrLivArea', 'YearBuilt'), ('GarageCars', 'GrLivArea', 'GrLivArea'), ('BsmtFinSF1', 'KitchenQual_Ex', 'LotArea'), ('BsmtExposure_Gd', 'OverallCond', 'YearRemodAdd'), ('GrLivArea', 'KitchenQual_Ex', 'OverallQual'), ('BsmtFinSF1', 'GarageArea', 'LotArea'), ('BsmtExposure_Gd', 'GrLivArea', 'YearRemodAdd'), ('BsmtFinSF1', 'BsmtQual_Ex', 'GrLivArea'), ('BsmtFinSF1', 'GarageArea', 'Neighborhood_Crawfor'), ('BsmtFinSF1', 'BsmtFinSF1', 'GarageArea'), ('BsmtFinSF1', 'OverallCond', 'TotalBsmtSF'), ('GrLivArea', 'KitchenAbvGr', 'LotArea'), ('GarageYrBlt', 'GrLivArea', 'LotArea'), ('KitchenQual_Ex', 'MSZoning_C (all)', 'WoodDeckSF'), ('GrLivArea', 'Neighborhood_OldTown', 'OverallCond'), ('BsmtExposure_No', 'GrLivArea', 'LotArea'), ('KitchenQual_Ex', 'SaleCondition_Family', 'WoodDeckSF'), ('Electrical_SBrkr', 'GrLivArea', 'LotArea'), ('GarageArea', 'GrLivArea', 'OverallCond'), ('BsmtExposure_Gd', 'LotArea', 'MSZoning_C (all)'), ('BsmtFinSF1', 'GrLivArea', 'TotalBsmtSF'), ('LotFrontage', 'MoSold_2', 'SaleType_New'), ('GarageYrBlt', 'GrLivArea', 'KitchenQual_Ex'), ('BsmtFinSF1', 'LotArea', 'YearRemodAdd'), ('BsmtExposure_Gd', 'BsmtFinSF1', 'LotArea'), ('BsmtFullBath', 'LotArea', 'Neighborhood_Crawfor'), ('1stFlrSF', 'BsmtExposure_Gd', 'YearBuilt'), ('1stFlrSF', 'BsmtExposure_Gd', 'LotFrontage'), ('BsmtFinSF1', 'HalfBath', 'HeatingQC_Ex'), ('1stFlrSF', 'KitchenQual_TA', 'LotArea'), ('BsmtFinSF1', 'GrLivArea', 'Neighborhood_Sawyer'), ('KitchenQual_TA', 'OverallCond', 'YearBuilt'), ('BsmtFinSF1', 'Functional_Typ', 'SaleType_New'), ('1stFlrSF', 'MoSold_2', 'SaleType_New'), ('BsmtFullBath', 'GarageYrBlt', 'OverallCond'), ('BsmtFinSF1', 'LotArea', 'SaleType_New'), ('BsmtExposure_Gd', 'Condition1_PosA', 'LotFrontage'), ('BsmtExposure_No', 'LotArea', 'TotRmsAbvGrd'), ('GrLivArea', 'LotArea', 'LotShape_Reg'), ('LotArea', 'MSSubClass_50', 'MSZoning_C (all)'), ('GrLivArea', 'LotShape_Reg', 'SaleCondition_Abnorml'), ('BsmtExposure_No', 'LotArea', 'SaleCondition_Abnorml'), ('SaleCondition_Abnorml', 'YearRemodAdd', 'YrSold_2009'), ('1stFlrSF', 'BsmtFinSF1', 'KitchenQual_TA'), ('GarageYrBlt', 'GrLivArea', 'OverallCond'), ('Neighborhood_Crawfor', 'OverallCond', 'YearBuilt'), ('BsmtExposure_Gd', 'Neighborhood_Crawfor', 'YearBuilt'), ('1stFlrSF', 'BsmtFinSF1', 'Neighborhood_Sawyer'), ('BsmtFinSF1', 'GrLivArea', 'HalfBath')]
data = data[depth1].copy()

test = test[depth1].copy()

for (a,b) in depth2[0:8]:

    data.loc[:,a+'_+_'+b] = data[a].add(data[b])

    data.loc[:,a+'_*_'+b] = data[a].mul(data[b])

    data.loc[:,a+'_-_'+b] = data[a].sub(data[b])

    #data.loc[:,a+'_abs(-)_'+b] = np.abs(data[a].sub(data[b]))

    data.loc[:,a+'_max_'+b] = np.maximum(data[a],data[b])

    data.loc[:,a+'_min_'+b] = np.minimum(data[a],data[b])

        

    test.loc[:,a+'_+_'+b] = test[a].add(test[b])

    test.loc[:,a+'_*_'+b] = test[a].mul(test[b])

    test.loc[:,a+'_-_'+b] = test[a].sub(test[b])

    #test.loc[:,a+'_abs(-)_'+b] = np.abs(test[a].sub(test[b]))

    test.loc[:,a+'_max_'+b] = np.maximum(test[a],test[b])

    test.loc[:,a+'_min_'+b] = np.minimum(test[a],test[b])

    

for (a,b,c) in depth3[0:10]:

    data.loc[:,a+'_+_'+b+'_+_'+c] = data[a].add(data[b]).add(data[c])

    test.loc[:,a+'_+_'+b+'_+_'+c] = test[a].add(test[b]).add(test[c])

    data.loc[:,a+'_*_'+b+'_*_'+c] = data[a].mul(data[b]).mul(data[c])

    test.loc[:,a+'_*_'+b+'_*_'+c] = test[a].mul(test[b]).mul(test[c])

    data.loc[:,a+'_+_'+b+'_*_'+c] = data[a].add(data[b]).mul(data[c])

    test.loc[:,a+'_+_'+b+'_*_'+c] = test[a].add(test[b]).mul(test[c])

    data.loc[:,a+'_*_'+b+'_+_'+c] = data[a].mul(data[b]).add(data[c])

    test.loc[:,a+'_*_'+b+'_+_'+c] = test[a].mul(test[b]).add(test[c])

    

    test.loc[:,a+'_-_'+b+'_+_'+c] = np.abs(test[a].sub(test[b])).add(test[c])

    test.loc[:,a+'_-_'+b+'_*_'+c] = np.abs(test[a].sub(test[b])).mul(test[c])

    test.loc[:,a+'_*_'+b+'_-_'+c] = test[a].mul(np.abs((test[b]).sub(test[c])))

    test.loc[:,a+'_+_'+b+'_-_'+c] = test[a].add(np.abs((test[b]).sub(test[c])))

    test.loc[:,a+'_-_'+b+'_-_'+c] = np.abs(np.abs(test[a].sub(test[b])).sub(test[c]))

    

    data.loc[:,a+'_-_'+b+'_+_'+c] = np.abs(data[a].sub(data[b])).add(data[c])

    data.loc[:,a+'_-_'+b+'_*_'+c] = np.abs(data[a].sub(data[b])).mul(data[c])

    data.loc[:,a+'_*_'+b+'_-_'+c] = data[a].mul(np.abs((data[b]).sub(data[c])))

    data.loc[:,a+'_+_'+b+'_-_'+c] = data[a].add(np.abs((data[b]).sub(data[c])))

    data.loc[:,a+'_-_'+b+'_-_'+c] = np.abs(np.abs(data[a].sub(data[b])).sub(data[c]))
print(data.shape)

print(test.shape)
for col in test.columns:

    if test[col].isna().any():

        print(col)
from  itertools import combinations



cc = list(combinations(data.columns,2))

data_1 = [(c[0],c[1],abs(data[c[1]].corr(data[c[0]]))) for c in cc]
to_drop = []

for (c1,c2,score) in data_1:

    if score > 0.95:

        to_drop.append(c2)

data.drop(to_drop, axis=1, inplace = True)

test.drop(to_drop, axis=1, inplace = True)
print(data.shape)

print(test.shape)
from sklearn.model_selection import KFold, cross_val_score

n_folds = 5



def mqe_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(data.values)

    mqe= cross_val_score(model, data.values,  y, scoring="neg_mean_absolute_error", cv = kf)

    return(mqe)
import xgboost as xgb

model_xgb = xgb.XGBRegressor(alpha= 0, colsample_bytree= 0.2, eta= 0.005, reg_lambda= 0, max_depth= 4, min_child_weight= 0, n_estimators= 500, random_state=2)
#XGB Parameter Tuning

gridParams = {

    'eta': [0.005],

    'max_depth': [4],

    'n_estimators': [500,1000],

    'min_child_weight' : [0],

    'colsample_bytree' : [0.2],

    'lambda' : [0],

    'alpha': [0],

    'gamma': [0]

    }

#grid = GridSearchCV(model_xgb, gridParams,verbose=1, cv=4,n_jobs=2, scoring = 'neg_mean_absolute_error')

# Run the grid

#grid.fit(data, y)



# Print the best parameters found

#print(grid.best_params_)

#print(abs(grid.best_score_))
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(feature_fraction= 0.3, learning_rate= 0.005, max_depth= 4, n_estimators= 5000, num_leaves= 8, random_state=2)  
#lgb Parameter Tuning

gridParams = {

    'num_leaves': [8],

    'max_depth': [4],

    'n_estimators': [5000],

    'feature_fraction' : [0.3],

    'learning_rate' : [0.005]

    }

#grid = GridSearchCV(model_lgb, gridParams,verbose=1, cv=4,n_jobs=2,scoring = 'neg_mean_absolute_error')

# Run the grid

#grid.fit(data, y)



# Print the best parameters found

#print(grid.best_params_)

#print(abs(grid.best_score_))
from sklearn.linear_model import ElasticNet

eNet = ElasticNet(alpha=0,l1_ratio=0,max_iter=60, random_state=2)
#ElasticNEt Parameter Tuning

gridParams = {

            'alpha': [0],

            'max_iter':[40,50,60,70,100,150,180],

            'l1_ratio': [0]

}

#grid = GridSearchCV(eNet, gridParams,verbose=1, cv=4,n_jobs=2,scoring = 'neg_mean_absolute_error')

# Run the grid

#grid.fit(data, y)



# Print the best parameters found

#print(grid.best_params_)

#print(abs(grid.best_score_))

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)
averaged_models = AveragingModels(models = (model_xgb,model_lgb, eNet))

score = mqe_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(abs(score.mean()),score.std()))
averaged_models.fit(data.values,y)

predictions =  (averaged_models.predict(test.values))#np.expm1
# The lines below shows how to save predictions in format used for competition scoring

output = pd.DataFrame({'Id': testId,

                      'SalePrice': predictions})

output.to_csv('submission.csv', index=False)