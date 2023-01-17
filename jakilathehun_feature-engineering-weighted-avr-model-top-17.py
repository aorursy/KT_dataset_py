# Main Libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# import operator for dictionary sorting operations
import operator

# preprocessing imports
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder

# train-test split
from sklearn.model_selection import train_test_split

# linear regression models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor

# 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

# stacking
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

# ignore warnings
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
warnings.filterwarnings("ignore", category=Warning)
print('Warnings will be ignored!')

# suppress scientific notation
pd.set_option('display.float_format', lambda x: '%.5f' % x)
# create dataframes for exploration
orig = pd.read_csv('../input/train.csv')
test_orig = pd.read_csv('../input/test.csv')
orig.head(20)
# quick glance at columns
orig.info()
# check out the nulls
print(orig.isnull().sum().sort_values(ascending=False).to_string())
print(test_orig.isnull().sum().sort_values(ascending=False).to_string())
# print value counts for all 'objects' with more than 1 null value
def object_vcs_and_nulls(df):
  for i in df:
    if df[i].dtype == 'O':
      if df[i].isnull().sum() > 0:
        print(df[i].value_counts())  
        print("Number of Null Values: " + str(df[i].isnull().sum()))
        print("Percentage of Nulls = " + str(np.round((df[i].isnull().sum() / 14.60), 2)) + "%")
        print("\n")
      
object_vcs_and_nulls(orig)
def mean_prices_categorical_var(df):
  for i in df:
    if df[i].dtype == 'O':
      print("=" * 20)
      print(orig.groupby(i, as_index=True)['SalePrice'].mean() )  
      print("\n")
      print(df[i].value_counts()) 
      print("\n")
      
print(mean_prices_categorical_var(orig))
# import data sets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# drop houses w/ over 4000 sq ft of GrLivArea
train = train.drop(train[train["GrLivArea"] > 4000].index, inplace=False)
# combine dataframes for simplicity
full = pd.concat([train,test], ignore_index=True)

# fix a few incorrect values; these values either had typos or had 'garage built' after house was sold
full['GarageYrBlt'][2588] = 2007
full['GarageYrBlt'][2545] = 2007
full['YearBuilt'][2545] = 2007
def feature_transformations(df):
  
  # drop nulls
  df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0)
  df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(0)
  df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(0)
  df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
  df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
  df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)
  df['GarageCars'] = df['GarageCars'].fillna(0)
  df['GarageArea'] = df['GarageArea'].fillna(0) 
  
  # feature transformations
  df['TotalLivingSF'] = df['GrLivArea'] + df['TotalBsmtSF'] - df['LowQualFinSF']
  df['TotalMainBath'] = df['FullBath'] + (df['HalfBath'] * 0.5)
  df['TotalBath'] = df['TotalMainBath'] + df['BsmtFullBath'] + (df['BsmtHalfBath'] * 0.5)
  df['AgeSold'] = df['YrSold'] - df['YearBuilt']
  df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
  df['TotalSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF']
  df['TotalArea'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GarageArea']
  
  # garage year built nulls and transformation
  df['GarageYrBlt'] = df['GarageYrBlt'].replace(np.nan, 1900)
  df['GarageAgeSold'] = df['YrSold'] - df['GarageYrBlt']
  
  # other age
  df['LastRemodelYrs'] = df['YrSold'] - df['YearRemodAdd']
  df['LastRemodelYrs'] = df['LastRemodelYrs'].replace(-1, 0)
  df['LastRemodelYrs'] = df['LastRemodelYrs'].replace(-2, 0)
  
  
  return df
  
full = feature_transformations(full)
train = feature_transformations(train)
def mean_price_map(feature):
  
  le = LabelEncoder()    
  feature = le.fit_transform(feature)  
  mean_prices = train.groupby(feature, as_index=True)['SalePrice'].mean()
  mean_price_length = len(mean_prices)
  numbers = np.linspace(0, mean_price_length, (mean_price_length+1))
  mean_price_dict = dict(zip(numbers, mean_prices))
  return mean_price_dict
    
  
def median_price_map(feature):
  
  le = LabelEncoder()    
  feature = le.fit_transform(feature)  
  med_prices = train.groupby(feature, as_index=True)['SalePrice'].median()
  med_price_length = len(med_prices)
  numbers = np.linspace(0, med_price_length, (med_price_length+1))
  med_price_dict = dict(zip(numbers, med_prices))
  return med_price_dict


def mean_square_footage(feature):
  
  le = LabelEncoder()    
  feature = le.fit_transform(feature)  
  mean_sqft = train.groupby(feature, as_index=True)['TotalLivingSF'].mean()
  mean_sqft_length = len(mean_sqft)
  numbers = np.linspace(0, mean_sqft_length, (mean_sqft_length+1))
  mean_sqft_dict = dict(zip(numbers, mean_sqft))
  return mean_sqft_dict
def df_transform(df):
  
  # neighborhood
  le = LabelEncoder()
  df['Neighborhood'] = le.fit_transform(df['Neighborhood'])
  df['NhoodMedianPrice'] = df['Neighborhood'].map(median_price_map(train['Neighborhood']))
  df['NhoodMeanPrice'] = df['Neighborhood'].map(mean_price_map(train['Neighborhood']))
  df['NhoodMeanSF'] = df['Neighborhood'].map(mean_square_footage(train['Neighborhood']))
  df['NhoodMeanPricePerSF'] = df['NhoodMeanPrice'] / df['NhoodMeanSF']
  df['ProxyPrice'] = df['NhoodMeanPricePerSF'] * df['TotalSF']

# fixer upper score
  df['FxUp_SaleCond'] = df.SaleCondition.map({'Partial': 0, 'Normal': 0, 'Alloca': 0, 'Family': 0, 
                                              'Abnorml': 3, 'AdjLand': 0, np.nan: 0}).astype(int)
  df['FxUp_Foundation'] = df.Foundation.map({'PConc':0, 'Wood':0, 'Stone':0, 'CBlock':1, 'BrkTil': 0, 'Slab': 2, np.nan: 0}).astype(int)
  df['FxUp_HeatingQC'] = df.HeatingQC.map({'Ex':0, 'Gd':0, 'TA':0, 'Fa':2, 'Po': 5, np.nan: 0}).astype(int)
  df['FxUp_Heating'] = df.Heating.map({'GasA':0, 'GasW':0, 'OthW':2, 'Wall':3, 'Grav': 4, 'Floor': 4, np.nan: 0}).astype(int)
  df['FxUp_CentralAir'] = df.CentralAir.map({'Y':0, 'N':6, np.nan: 0}).astype(int)
  df['FxUp_GarageQual'] = df.GarageQual.map({'Ex':0, 'Gd':0, 'TA':0, 'Fa':1, 'Po': 3, np.nan: 0}).astype(int)
  df['FxUp_PavedDrive'] = df['PavedDrive'].map({'Y':0, 'P':0, 'N':2, np.nan: 0}).astype(int)
  df['FxUp_Electrical'] = df.Electrical.map({'SBrkr':0, 'FuseA':2, 'FuseF':2, 'FuseP':2, 'Mix': 4, np.nan: 0}).astype(int)
  df['FxUp_MSZoning'] = df.MSZoning.map({'FV':0, 'RL':0, 'RM':0, 'RH':0, 'C (all)':3 , np.nan: 0}).astype(int)
  df['FxUp_Street'] = df.Street.map({'Pave':0, 'Grvl':3, np.nan: 0}).astype(int)
  df['FxUp_OverallQual'] = df.OverallQual.map({1: 5, 2: 5, 3: 3, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0})
  df['FxUp_KitchenQual']= df.KitchenQual.map({'Ex':0, 'Gd':0, 'TA':0, 'Fa':1, 'Po': 4, np.nan: 0}).astype(int)
  
  df['FixerUpperScore'] = (45 - df['FxUp_SaleCond'] - df['FxUp_Foundation'] - df['FxUp_HeatingQC'] 
                           - df['FxUp_Heating'] - df['FxUp_CentralAir'] - df['FxUp_GarageQual'] - df['FxUp_PavedDrive'] -
                           df['FxUp_Electrical'] - df['FxUp_MSZoning'] - df['FxUp_Street'] - df['FxUp_OverallQual'] - df['FxUp_KitchenQual'])
  
  
  # map MSSubClass
  df['MSSubClass'] = df['MSSubClass'].astype(str)
  df['MSSubClass'] = df.MSSubClass.map({'180':1, '30':2, '45':2, '190':3, '50':3, '90':3, 
                                        '85':4, '40':4, '160':4, '70':5, '20':5, '75':5, '80':5, '150':5,
                                        '120': 6, '60':6})
  
  # LotAreaCut
  df["LotAreaCut"] = pd.qcut(df.LotArea,10)
  df["LotAreaCut"] = df.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
  

 
  
  # drop cfeatures  
  # drop 'RoofMat1' as 98% of values are 'CompShq' and other values have too small sample size
  # drop Exterior2nd as problematic in df and not necessarily as important
  df = df.drop(['RoofMatl', 'Exterior2nd'], axis=1)
  
  
  # assign labels for alley
  df['Alley'] = df['Alley'].replace('Pave', 1)
  df['Alley'] = df['Alley'].replace('Grvl', 0)
  df['Alley'] = df['Alley'].replace(np.nan, 2)
  df['Alley'] = df['Alley'].astype(int)
  
  # assign labels for MasVnrType
  df['MasVnrType'] = df['MasVnrType'].replace('Stone', 4)
  df['MasVnrType'] = df['MasVnrType'].replace('BrkFace', 3)
  df['MasVnrType'] = df['MasVnrType'].replace('BrkCmn', 2)
  df['MasVnrType'] = df['MasVnrType'].replace('CBlock', 1)
  df['MasVnrType'] = df['MasVnrType'].replace('None', 0)
  df['MasVnrType'] = df['MasVnrType'].replace(np.nan, 0)
  df['MasVnrType'] = df['MasVnrType'].astype(int)
  
  # masonry veneer area
  df['MasVnrArea'] = df['MasVnrArea'].replace(np.nan, 0)
#   df['MasVnrArea'] = df['MasVnrArea'].astype(int)
  
  # assign value labels for basement features
  df['BsmtQual'] = df.BsmtQual.map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po': 1, np.nan: 0})
  df['BsmtQual'] = df['BsmtQual'].astype(int)
  
  df['BsmtCond'] = df.BsmtCond.map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po': 1, np.nan: 0})
  df['BsmtCond'] = df['BsmtCond'].astype(int)
  
  df['BsmtExposure'] = df.BsmtExposure.map({'Gd':4, 'Av':3, 'Mn':2, 'No': 1, np.nan: 0})
  df['BsmtExposure'] = df['BsmtExposure'].astype(int)
  
  df['BsmtFinType1'] = df.BsmtFinType1.map({'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ': 2, 'Unf': 1, np.nan: 0})
  df['BsmtFinType1'] = df['BsmtFinType1'].astype(int)
  
  df['BsmtFinType2'] = df.BsmtFinType2.map({'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ': 2, 'Unf': 1, np.nan: 0})
  df['BsmtFinType2'] = df['BsmtFinType2'].astype(int)  
  
  # electrical mapping; replace nulls with "3" for standard breaker (mode)
  df['Electrical'] = df.Electrical.map({'SBrkr':3, 'FuseA':2, 'FuseF':1, 'FuseP':0, 'Mix': 1, np.nan: 3})
  df['Electrical'] = df['Electrical'].astype(int)
  
  # fireplace mapping
  df['FireplaceQu'] = df.FireplaceQu.map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po': 1, np.nan: 0})
  df['FireplaceQu'] = df['FireplaceQu'].astype(int)
  
  # garage features
  df['GarageType'] = df.GarageType.map({'2Types':5, 'Attchd':4, 'Basment':3, 'BuiltIn':4, 'CarPort': 1, 'Detchd': 2,np.nan: 0})
  df['GarageType'] = df['GarageType'].astype(int)
  
  df['GarageFinish'] = df.GarageFinish.map({'Fin':3, 'RFn':2, 'Unf':1, np.nan: 0})
  df['GarageFinish'] = df['GarageFinish'].astype(int)
  
  df['GarageQual'] = df.GarageQual.map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po': 1, np.nan: 0})
  df['GarageQual'] = df['GarageQual'].astype(int)
  
  df['GarageCond'] = df.GarageCond.map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po': 1, np.nan: 0})
  df['GarageCond'] = df['GarageCond'].astype(int)
  
  # miscellenous feature mapping
  df['PoolQC'] = df.PoolQC.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, np.nan: 0})
  df['PoolQC'] = df['PoolQC'].astype(int)
  
  df['Fence'] = df.Fence.map({'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1, np.nan: 0})
  df['Fence'] = df['Fence'].astype(int)
  
  df['MiscFeature'] = df.MiscFeature.map({'Shed':1, 'Elev':0, 'Gar2':0, 'Othr':0, 'TenC':0, np.nan: 0})
  df['MiscFeature'] = df['MiscFeature'].astype(int)
  df['Shed'] = df['MiscFeature']
  df = df.drop(['MiscFeature'], axis=1)
  

  
  # fill in remaining nulls
  df['LotFrontage'] = df['LotFrontage'].replace(np.nan, 0)
  
  # deal with categorial variables
  df['MSZoning'] = df.MSZoning.map({'FV':4, 'RL':3, 'RM':2, 'RH':2, 'C (all)':1 , np.nan: 3})
  df['MSZoning'] = df['MSZoning'].astype(int)
  
  df['Street'] = df.Street.map({'Pave':1, 'Grvl':0, np.nan: 1}) 
  df['Street'] = df['Street'].astype(int)
  
  # assign value of 0 to regular lots; 1 for all categories of irregular
  df['LotShape'] = df.LotShape.map({'Reg':0, 'IR1':1, 'IR2':1, 'IR3':1, np.nan: 1}) 
  df['LotShape'] = df['LotShape'].astype(int)
  
  # assign value of '3' to hillside, '2' to level, low, and nulls, '1' to banked
  df['LandContour'] = df.LandContour.map({'HLS':3, 'Bnk':1, 'Lvl':2, 'Low':2, np.nan: 2}) 
  df['LandContour'] = df['LandContour'].astype(int)
  
  # only 1 entry w/ public utilities
  df['Utilities'] = df.Utilities.map({'AllPub':1, 'NoSeWa':0, np.nan: 2}) 
  df['Utilities'] = df['Utilities'].astype(int)
  
  # mode = inside
  df['LotConfig'] = df.LotConfig.map({'CulDSac':2, 'FR3':1, 'FR2':1, 'Corner':1, 'Inside':0, np.nan: 0}) 
  df['LotConfig'] = df['LotConfig'].astype(int)
  
  # land slope, mode = Gtl
  df['LandSlope'] = df.LandSlope.map({'Sev':2, 'Mod':1, 'Gtl':0, np.nan: 0}) 
  df['LandSlope'] = df['LandSlope'].astype(int)
  
  # proxmmity to conditions
  df['Condition1'] = df.Condition1.map({'PosA':5, 'PosN':4, 'RRNe':3, 'RRNn':3, 
                                        'Norm':2, 'Feedr':0, 'Artery':0, 'RRAn':1, 'RRAe':0, np.nan: 2}) 
  df['Condition1'] = df['Condition1'].astype(int)
  
  df['Condition2'] = df.Condition1.map({'PosA':5, 'PosN':4, 'RRNe':3, 'RRNn':3, 
                                        'Norm':2, 'Feedr':0, 'Artery':0, 'RRAn':1, 'RRAe':0, np.nan: 2}) 
  df['Condition2'] = df['Condition1'].astype(int)

  # 
  df['BldgType'] = df.BldgType.map({'1Fam':4, 'TwnhsE':3, 'Twnhs':2, 'Duplex':1, '2fmCon':0, np.nan: 4}) 
  df['BldgType'] = df['BldgType'].astype(int)
  
  df['HouseStyle'] = df.HouseStyle.map({'2.5Fin':7, '2Story':6, '1Story':5, 'SLvl':4, 
                                        '2.5Unf':3, '1.5Fin':2, 'SFoyer':1, '1.5Unf':0, np.nan: 5}) 
  df['HouseStyle'] = df['HouseStyle'].astype(int)
  
  # gabel and hip most common roof styles by far; guess on value of others
  df['RoofStyle'] = df.RoofStyle.map({'Hip':2, 'Shed':2, 'Gable':1, 'Mansard':1, 'Flat':1, 'Gambrel':0, np.nan: 1}) 
  df['RoofStyle'] = df['RoofStyle'].astype(int)
  
  df['Exterior1st'] = df.Exterior1st.map({'Stone': 8, 'CemntBd': 7, 'VinylSd': 6, 'BrkFace': 5, 
                                        'Plywood': 4, 'HdBoard': 3, 'Stucco': 2, 'ImStucc': 2, 
                                        'WdShing': 1, 'Wd Sdng': 1, 'MetalSd': 1, 'BrkComm': 0, 
                                        'CBlock': 0, 'AsphShn': 0, 'AsbShng': 0, np.nan: 3}) 
  df['Exterior1st'] = df['Exterior1st'].astype(int)
  
    
#   df['Exterior2nd'] = df.Exterior2nd.map({'Stone': 8, 'CmntBd': 7, 'VinylSd': 6, 'BrkFace': 5, 
#                                         'Plywood': 4, 'HdBoard': 3, 'Stucco': 2, 'ImStucc': 2, 
#                                         'Wd Shng': 1, 'Wd Sdng': 1, 'MetalSd': 1, 'Brk Cmn': 0, 
#                                         'CBlock': 0, 'AsphShn': 0, 'AsbShng': 0, 'Other': 3, np.nan: 3}) 
#   df['Exterior2nd'] = df['Exterior2nd'].astype(int)
  
  df['ExterQual'] = df.ExterQual.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, np.nan: 1})
  df['ExterQual'] = df['ExterQual'].astype(int)
  
  df['ExterCond'] = df.ExterCond.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po': 0, np.nan: 1})
  df['ExterCond'] = df['ExterCond'].astype(int)
  
  df['Foundation'] = df.Foundation.map({'PConc':3, 'Wood':2, 'Stone':2, 'CBlock':2, 'BrkTil': 1, 'Slab': 0, np.nan: 2})
  df['Foundation'] = df['Foundation'].astype(int)
  
  df['Heating'] = df.Heating.map({'GasA':2, 'GasW':1, 'OthW':0, 'Wall':0, 'Grav': 0, 'Floor': 0, np.nan: 2})
  df['Heating'] = df['Heating'].astype(int)
  
  df['HeatingQC'] = df.HeatingQC.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po': 0, np.nan: 1})
  df['HeatingQC'] = df['HeatingQC'].astype(int)
  
  df['CentralAir'] = df.CentralAir.map({'Y':1, 'N':0, np.nan: 1})
  df['CentralAir'] = df['CentralAir'].astype(int)
  
  df['KitchenQual'] = df.KitchenQual.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po': 0, np.nan: 2})
  df['KitchenQual'] = df['KitchenQual'].astype(int)
  
  df['Functional'] = df.Functional.map({'Typ':7, 'Min1':6, 'Min2':5, 'Mod':4, 
                                        'Maj1':3, 'Maj2':2, 'Sev':1, 'Sal':0, np.nan: 7}) 
  df['Functional'] = df['Functional'].astype(int)
  
  df['PavedDrive'] = df['PavedDrive'].map({'Y':2, 'P':1, 'N':0, np.nan: 2})
  df['PavedDrive'] = df['PavedDrive'].astype(int)
  
  df['SaleType'] = df.SaleType.map({'New': 2, 'WD': 1, 'CWD': 1, 'Con': 1, 'ConLI': 1, 
                                        'ConLD':1, 'ConLw':1, 'COD': 0, 'Oth': 0, np.nan: 1}) 
  df['SaleType'] = df['SaleType'].astype(int)
  
  df['SaleCondition'] = df.SaleCondition.map({'Partial': 5, 'Normal': 4, 'Alloca': 4, 
                                              'Family': 2, 'Abnorml': 1, 'AdjLand': 0, np.nan: 4})
  df['SaleCondition'] = df['SaleCondition'].astype(int)
  
  df['SeasonSold'] = df.MoSold.map({1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:0})
  df['SeasonSold'] = df['SeasonSold'].astype(int)
  
  
#   # Rashid transformations
  df['TotalHouseOverallQual'] = df['TotalSF'] * df['OverallQual']
#   df['ZoningPrice'] = df['MSSubCl_MeanPPSF'] * df['TotalSF']
  df['Functional_OverallQual'] = df['Functional'] * df['OverallQual']
  df['TotalSF_LotArea'] = df['TotalSF'] * df['LotArea']
  df['TotalSF_Condition'] = df['TotalSF'] * df['Condition1']



   
  return df
full = df_transform(full)
n_train=train.shape[0]
df = full[:n_train]
explore1 = df[['SalePrice', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'Exterior1st', 'MasVnrType', 'MasVnrArea', 'ExterQual',
       'ExterCond']]

explore2 = df[['SalePrice', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
       'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
       'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'KitchenQual', 'TotRmsAbvGrd']]

explore3 = df[['SalePrice', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscVal', 'MoSold',
       'YrSold', 'SaleType', 'SaleCondition', 'Shed']]

explore4 = df[['SalePrice', 'NhoodMedianPrice', 'PoolArea', 'PoolQC', 'Fence', 'MiscVal', 'MoSold',
       'YrSold', 'SaleType', 'SaleCondition', 'Shed']]
corr1 = explore1.corr()
plt.figure(figsize = (16,12))
sns.heatmap(corr1, vmin=-1, vmax=1, 
            xticklabels=corr1.columns.values,
            yticklabels=corr1.columns.values, cmap='Spectral_r')
corr2 = explore2.corr()
plt.figure(figsize = (16,12))
sns.heatmap(corr2, vmin=-1, vmax=1, 
            xticklabels=corr2.columns.values,
            yticklabels=corr2.columns.values, cmap='Spectral_r')
corr3 = explore3.corr()
plt.figure(figsize = (16,12))
sns.heatmap(corr3, vmin=-1, vmax=1, 
            xticklabels=corr3.columns.values,
            yticklabels=corr3.columns.values, cmap='Spectral_r')
corr4 = explore4.corr()
plt.figure(figsize = (16,12))
sns.heatmap(corr4, vmin=-1, vmax=1, 
            xticklabels=corr4.columns.values,
            yticklabels=corr4.columns.values, cmap='Spectral_r')
corr_list = sorted(df.corr().to_dict()['SalePrice'].items(), key=lambda x: x[1], reverse=True)
corr_list

from scipy.stats import skew

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in full.drop(['FxUp_SaleCond'], axis=1):
    if full[i].dtype in numeric_dtypes: 
        numerics2.append(i)

skew_features = full[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
skews = pd.DataFrame({'skew':skew_features})
skews = skews.drop(['SalePrice'], axis=0)
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

high_skew = skew_features[skew_features > 0.75]
skew_index = high_skew.index
   
for i in high_skew.index:
    full[i]= boxcox1p(full[i], boxcox_normmax(full[i]+1))
        
skew_features2 = full[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
skews2 = pd.DataFrame({'skew':skew_features2})
print(skews2.to_string())
# cut out a few features that seemed unuseful
full_condensed = full[['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'Alley', 'BedroomAbvGr',
       'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath',
       'BsmtQual', 'BsmtUnfSF', 'CentralAir', 'Condition1', 'Condition2',
       'Electrical', 'EnclosedPorch', 'ExterCond', 'ExterQual', 'Exterior1st',
       'Fence', 'FireplaceQu', 'Fireplaces', 'Foundation', 'FullBath',
       'Functional', 'GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish',
       'GarageQual', 'GarageType', 'GrLivArea', 'HalfBath',
       'Heating', 'HeatingQC', 'HouseStyle', 'KitchenAbvGr',
       'KitchenQual', 'LandContour', 'LandSlope', 'LotArea', 'LotConfig', 'LotFrontage',
       'LotShape', 'LowQualFinSF', 'MSSubClass', 'MSZoning', 'MasVnrArea',
       'MasVnrType', 'MiscVal', 'MoSold', 'OpenPorchSF',
       'OverallCond', 'OverallQual', 'PavedDrive', 'PoolArea', 'PoolQC',
       'RoofStyle', 'SaleCondition', 'SalePrice', 'SaleType', 'ScreenPorch',
       'Street', 'TotRmsAbvGrd', 'TotalBsmtSF', 'Utilities', 'WoodDeckSF',
       'TotalLivingSF', 'TotalMainBath',
       'TotalBath', 'AgeSold', 'TotalPorchSF', 'TotalSF', 'TotalArea',
       'GarageAgeSold', 'LastRemodelYrs', 'NhoodMedianPrice', 'NhoodMeanPrice',
       'NhoodMeanSF', 'NhoodMeanPricePerSF', 'ProxyPrice', 
       'FixerUpperScore', 'LotAreaCut', 'Shed', 'SeasonSold']]

n_train=train.shape[0]

train = full_condensed[:n_train]
test = full_condensed[n_train:]

scaler = RobustScaler()

X = train.drop(['SalePrice'], axis=1)
test_X = test.drop(['SalePrice'], axis=1)

y = train['SalePrice']

X_scaled = scaler.fit(X).transform(X)
y_log = np.log(train['SalePrice'])
test_X_scaled = scaler.transform(test_X)

X_scaled_df = pd.DataFrame(X_scaled)
y_log_df = pd.DataFrame(y_log)
test_X_scaled_df = pd.DataFrame(test_X_scaled)
lasso_fi=Lasso(alpha=0.001)
lasso_fi.fit(X_scaled,y_log)
FI_lasso = pd.DataFrame({"Feature Importance":lasso_fi.coef_}, index=X.columns)
FI_sorted = FI_lasso.sort_values("Feature Importance",ascending=False)
print(FI_sorted.to_string())
FI_sorted.index

# Root mean squared error (RMSE)
def rmse(y_pred, y_test):
  return np.sqrt(mean_squared_error(y_test, y_pred))


class CvScore(object):
  def __init__(self, list, name_list, X, y, folds=5, score='neg_mean_squared_error', seed=66, split=0.33):
    self.X = X
    self.y = y
    self.folds = folds
    self.score = score
    self.seed = seed
    self.split = split
    self.model = list[0]
    self.list = list
    self.name = name_list[0]
    self.name_list = name_list
    
  def cv(self):
    cv_score = cross_val_score(self.model, self.X, self.y, cv=self.folds, scoring=self.score)
    score_array = np.sqrt(-cv_score)
    mean_rmse = np.mean(score_array)
    print("Mean RMSE: ", mean_rmse)
    
  def cv_list(self):
    for name, model in zip(self.name_list, self.list):
      cv_score = cross_val_score(model, self.X, self.y, cv=self.folds, scoring=self.score)
      score_array = np.sqrt(-cv_score)
      mean_rmse = np.mean(score_array)
      std_rmse = np.std(score_array)
      print("{}: {:.5f}, {:.4f}".format(name, mean_rmse, std_rmse))
lr = LinearRegression()
ridge = Ridge(alpha=40)
lasso = Lasso(alpha=0.0001, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=True, positive=False, precompute=False, random_state=66,
   selection='cyclic', tol=0.0001, warm_start=False)
lasso_lars = LassoLarsIC()


rfr = RandomForestRegressor()
etr = ExtraTreesRegressor(max_depth=40, max_features='auto', n_estimators=900)
gbr = GradientBoostingRegressor()
xgb = XGBRegressor(max_depth=2, n_estimators=400, colsample_bytree=0.5)

svr = SVR(C=1, gamma=0.00001, epsilon=0.0)
linear_svr = LinearSVR(C=10)

# en = ElasticNet(alpha=0.004, l1_ratio=0.3, max_iter=10000)
en = ElasticNet(alpha=0.005, l1_ratio=0.1, max_iter=1000)

br = BayesianRidge()
kr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


ada = AdaBoostRegressor(Lasso(alpha=0.0001, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=True, positive=False, precompute=False, random_state=66,
   selection='cyclic', tol=0.0001, warm_start=False))

regression_list = [lr, ridge, lasso, lasso_lars, rfr, etr, gbr, xgb, svr, linear_svr, en, br, kr]
name_list = ["Linear", "Ridge", "Lasso", "Lasso Lars", "Random Forest", "Extra Trees", "Grad Boost", "XGBoost", "SVR", "LinSVR", 
             "ElasticNet", "Bayesian Ridge", "Kernel Ridge"]
v1 = CvScore(regression_list, name_list, X_scaled, y_log)
v1.cv_list()
full_v2 = full[['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'Alley', 'BedroomAbvGr',
       'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath',
       'BsmtQual', 'BsmtUnfSF', 'CentralAir', 'Condition1', 'Condition2',
       'Electrical', 'EnclosedPorch', 'ExterCond', 'ExterQual', 'Exterior1st',
       'Fence', 'FireplaceQu', 'Fireplaces', 'Foundation', 'FullBath',
       'Functional', 'GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish',
       'GarageQual', 'GarageType', 'GrLivArea', 'HalfBath',
       'Heating', 'HeatingQC', 'HouseStyle', 'KitchenAbvGr',
       'KitchenQual', 'LandContour', 'LandSlope', 'LotArea', 'LotConfig', 'LotFrontage',
       'LotShape', 'LowQualFinSF', 'MSSubClass', 'MSZoning', 'MasVnrArea',
       'MasVnrType', 'MiscVal', 'MoSold', 'OpenPorchSF',
       'OverallCond', 'OverallQual', 'PavedDrive', 'PoolArea', 'PoolQC',
       'RoofStyle', 'SaleCondition', 'SalePrice', 'SaleType', 'ScreenPorch',
       'Street', 'TotRmsAbvGrd', 'TotalBsmtSF', 'Utilities', 'WoodDeckSF',
       'TotalLivingSF', 'TotalMainBath',
       'TotalBath', 'AgeSold', 'TotalPorchSF', 'TotalSF', 'TotalArea',
       'GarageAgeSold', 'LastRemodelYrs', 'NhoodMedianPrice', 'NhoodMeanPrice',
       'NhoodMeanSF', 'NhoodMeanPricePerSF',  
       'FixerUpperScore', 'LotAreaCut', 'Shed', 'SeasonSold']]

n_train=train.shape[0]

train_v2 = full_v2[:n_train]
test_v2 = full_v2[n_train:]

scaler = RobustScaler()

X_v2 = train_v2.drop(['SalePrice'], axis=1)
test_X_v2 = test_v2.drop(['SalePrice'], axis=1)
y_v2 = train_v2['SalePrice']

X_scaled_v2 = scaler.fit(X_v2).transform(X_v2)
y_log_v2 = np.log(train_v2['SalePrice'])
test_X_scaled_v2 = scaler.transform(test_X_v2)
v2 = CvScore(regression_list, name_list, X_scaled_v2, y_log_v2)
v2.cv_list()
full_v3 = full[['OverallQual', 'ProxyPrice', 'GrLivArea', '2ndFlrSF', '1stFlrSF',
       'TotalBsmtSF', 'OverallCond', 'NhoodMeanPricePerSF', 'MSZoning',
       'Functional', 'GarageCars', 'LotArea', 'KitchenQual', 'SaleType',
       'Fireplaces', 'Condition1', 'SaleCondition', 'HeatingQC', 'GarageArea',
       'BldgType', 'BsmtExposure', 'Exterior1st', 'TotalBath', 'WoodDeckSF',
       'ExterQual', 'Foundation', 'FixerUpperScore', 'TotRmsAbvGrd',
       'GarageFinish', 'LotConfig', 'RoofStyle', 'FireplaceQu', 'TotalPorchSF', 'SalePrice']]

n_train=train.shape[0]

train_v3 = full_v3[:n_train]
test_v3 = full_v3[n_train:]

scaler = RobustScaler()

X_v3 = train_v3.drop(['SalePrice'], axis=1)
test_X_v3 = test_v3.drop(['SalePrice'], axis=1)
y_v3 = train_v3['SalePrice']

X_scaled_v3 = scaler.fit(X_v3).transform(X_v3)
y_log_v3 = np.log(train_v3['SalePrice'])
test_X_scaled_v3 = scaler.transform(test_X_v3)
v3 = CvScore(regression_list, name_list, X_scaled_v3, y_log_v3)
v3.cv_list()
full_v4 = full[['OverallQual', 'ProxyPrice', 'GrLivArea', '2ndFlrSF', '1stFlrSF',
       'TotalBsmtSF', 'OverallCond', 'NhoodMeanPricePerSF', 'MSZoning',
       'Functional', 'GarageCars', 'LotArea', 'KitchenQual', 'SaleType',
       'Fireplaces', 'Condition1', 'SaleCondition', 'HeatingQC', 'GarageArea',
       'BldgType', 'BsmtExposure', 'Exterior1st', 'TotalBath', 'WoodDeckSF',
       'ExterQual', 'Foundation', 'FixerUpperScore', 'TotRmsAbvGrd',
       'GarageFinish', 'FireplaceQu', 'TotalPorchSF', 'SalePrice']]

n_train=train.shape[0]

train_v4 = full_v4[:n_train]
test_v4 = full_v4[n_train:]

scaler = RobustScaler()

X_v4 = train_v4.drop(['SalePrice'], axis=1)
test_X_v4 = test_v4.drop(['SalePrice'], axis=1)
y_v4 = train_v4['SalePrice']

X_scaled_v4 = scaler.fit(X_v4).transform(X_v4)
y_log_v4 = np.log(train_v4['SalePrice'])
test_X_scaled_v4 = scaler.transform(test_X_v4)
v4 = CvScore(regression_list, name_list, X_scaled_v4, y_log_v4)
v4.cv_list()
class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        print("Best parameters found: ", grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
# param_grid={'C':[0.01, 0.1, 1, 10, 100], 'gamma':[0.00001, 0.0001, .001, .01, .1], 'epsilon': np.linspace(0,1,10)}
# grid(SVR()).grid_get(X_scaled_v3,y_log_v3,param_grid)
# param_grid={'C':[0.01, 0.1, 0.25, 0.5, 0.75, 1, 5, 10, 100]}
# grid(LinearSVR()).grid_get(X_scaled_v3,y_log_v3,param_grid)
# param_grid={'alpha':[0.00001, 0.0001, 0.01, 0.1, 0.25, 0.5, 0.75, 1]}
# grid(Lasso()).grid_get(X_scaled_v3,y_log_v3,param_grid)
# param_grid={'alpha':np.linspace(0.2, 1.0, 9), 'kernel':["polynomial"], 'degree':[1,2,3],'coef0': np.linspace(2,10,9)}
# grid(KernelRidge()).grid_get(X_scaled_v3,y_log_v3,param_grid)
# # Extreme Gradient Boost
# xgbm_param_grid = {
#      'colsample_bytree': np.linspace(0.5, 1.0, 5),
#      'n_estimators':[200, 400, 600, 800, 1000],
#      'max_depth': [1, 2, 5, 10],
# }

# grid(XGBRegressor()).grid_get(X_scaled_v3,y_log_v3,xgbm_param_grid)
# # Random Forest Regression
# rfr_param_grid = {
#      'n_estimators':[100, 200, 400, 600, 800],
#      'max_depth': [80, 120, 160, 200]
# }

# grid(RandomForestRegressor()).grid_get(X_scaled_v3,y_log_v3,rfr_param_grid)
# # Extra Trees Regression
# etr_param_grid = {
#      'n_estimators':[400, 600, 800, 900, 1000],
#      'max_depth': [5, 10, 20, 40, 50, 60]
# }

# grid(ExtraTreesRegressor()).grid_get(X_scaled_v3,y_log_v3,etr_param_grid)
lr = LinearRegression()
ridge = Ridge(alpha=40)

# Best parameters found:  {'alpha': 0.0001} 0.11187841593433792
lasso = Lasso(alpha=0.0001)

lasso_lars = LassoLarsIC()

# Best parameters found:  {'max_depth': 120, 'n_estimators': 800} 0.13093022119435033
rfr = RandomForestRegressor(max_depth=120, n_estimators=800)

# Best parameters found:  {'max_depth': 60, 'n_estimators': 600} 0.12379414612361438
etr = ExtraTreesRegressor(max_depth=60, max_features='auto', n_estimators=600)

gbr = GradientBoostingRegressor()

# Best parameters found:  {'colsample_bytree': 0.5, 'max_depth': 2, 'n_estimators': 600}
xgb = XGBRegressor(max_depth=2, n_estimators=600, colsample_bytree=0.5)

# Best parameters found:  {'C': 10, 'epsilon': 0.0, 'gamma': 0.001} 0.10962793797503995
svr = SVR(C=10, gamma=0.001, epsilon=0.0)

# Best parameters found:  {'C': 0.75} 0.11327745125722458
linear_svr = LinearSVR(C=0.75)

# en = ElasticNet(alpha=0.004, l1_ratio=0.3, max_iter=10000)
en = ElasticNet(alpha=0.005, l1_ratio=0.1, max_iter=1000)

br = BayesianRidge()

# Best parameters found:  {'alpha': 1.0, 'coef0': 6.0, 'degree': 2, 'kernel': 'polynomial'} 0.11116062462422147
kr = KernelRidge(alpha=1.0, kernel='polynomial', degree=2, coef0=6.0)


regression_list = [lr, ridge, lasso, lasso_lars, rfr, etr, gbr, xgb, svr, linear_svr, en, br, kr]
name_list = ["Linear", "Ridge", "Lasso", "Lasso Lars", "Random Forest", "Extra Trees", "Grad Boost", "XGBoost", "SVR", "LinSVR", 
             "ElasticNet", "Bayesian Ridge", "Kernel Ridge"]
v3 = CvScore(regression_list, name_list, X_scaled_v3, y_log_v3)
v3.cv_list()
# define cross validation strategy
def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse


class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self,mod,weight):
        self.mod = mod
        self.weight = weight
        
    def fit(self,X,y):
        self.models_ = [clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X,y)
        return self
    
    def predict(self,X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        # for every data point, single model prediction times weight, then add them together
        for data in range(pred.shape[1]):
            single = [pred[model,data]*weight for model,weight in zip(range(pred.shape[0]),self.weight)]
            w.append(np.sum(single))
        return w
# even weighted model
var = 6

w1 = 1/var
w2 = 1/var
w3 = 1/var
w4 = 1/var
w5 = 1/var
w6 = 1/var

weight_avg = AverageWeight(mod = [en, lasso, ridge, xgb, svr, kr],weight=[w1,w2,w3,w4,w5,w6])

score = rmse_cv(weight_avg,X_scaled,y_log)
print(score.mean())
# even weighted model
var = 5

w1 = 1/var
w2 = 1/var
w3 = 1/var
w4 = 1/var
w5 = 1/var

weight_avg = AverageWeight(mod = [en, lasso, ridge, svr, kr],weight=[w1,w2,w3,w4,w5])

score = rmse_cv(weight_avg,X_scaled_v3,y_log_v3)
print(score.mean())
# weighted towards higher scoring models

w1 = .1
w2 = .25
w3 = .05
w4 = .1
w5 = .25
w6 = .25

weight_avg = AverageWeight(mod = [en, lasso, ridge, xgb, svr, kr],weight=[w1,w2,w3,w4,w5,w6])

score = rmse_cv(weight_avg,X_scaled_v3,y_log_v3)
print(score.mean())
class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,mod,meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)
        
    def fit(self,X,y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))
        
        for i,model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X,y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index,i] = renew_model.predict(X[val_index])
        
        self.meta_model.fit(oof_train,y)
        return self
    
    def predict(self,X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) 
                                      for single_model in self.saved_model]) 
        return self.meta_model.predict(whole_test)
    
    def get_oof(self,X,y,test_X):
        oof = np.zeros((X.shape[0],len(self.mod)))
        test_single = np.zeros((test_X.shape[0],5))
        test_mean = np.zeros((test_X.shape[0],len(self.mod)))
        for i,model in enumerate(self.mod):
            for j, (train_index,val_index) in enumerate(self.kf.split(X,y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index],y[train_index])
                oof[val_index,i] = clone_model.predict(X[val_index])
                test_single[:,j] = clone_model.predict(test_X)
            test_mean[:,i] = test_single.mean(axis=1)
        return oof, test_mean
a = Imputer().fit_transform(X_scaled)
b = Imputer().fit_transform(y_log.values.reshape(-1,1)).ravel()

stack_model = stacking(mod=[lasso, ridge, xgb, en, kr],meta_model=svr)

score = rmse_cv(stack_model,a,b)
print(score.mean())
# This is the final model I use; the 6-model even-weighted ensemble seemed to work best

# even weighted model
var = 6

w1 = 1/var
w2 = 1/var
w3 = 1/var
w4 = 1/var
w5 = 1/var
w6 = 1/var

final_model = AverageWeight(mod = [en, lasso, ridge, xgb, svr, kr],weight=[w1,w2,w3,w4,w5,w6])
final_model = final_model.fit(a,b)
pred = np.exp(final_model.predict(test_X_scaled))
print(pred)

id = test_orig['Id']
result=pd.DataFrame({'Id': id, 'SalePrice':pred})
result.to_csv("submission.csv",index=False)
full_v6 = full[['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'Alley', 'BedroomAbvGr',
       'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath',
       'BsmtQual', 'BsmtUnfSF', 'CentralAir', 'Condition1', 'Condition2',
       'Electrical', 'EnclosedPorch', 'ExterCond', 'ExterQual', 'Exterior1st',
       'Fence', 'FireplaceQu', 'Fireplaces', 'Foundation', 'FullBath',
       'Functional', 'GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish',
       'GarageQual', 'GarageType', 'GarageYrBlt', 'GrLivArea', 'HalfBath',
       'Heating', 'HeatingQC', 'HouseStyle', 'KitchenAbvGr',
       'KitchenQual', 'LandContour', 'LandSlope', 'LotArea', 'LotConfig',
       'LotFrontage', 'LotShape', 'LowQualFinSF', 'MSSubClass', 'MSZoning', 'MasVnrArea',
       'MasVnrType', 'MiscVal', 'MoSold', 'Neighborhood', 'OpenPorchSF',
       'OverallCond', 'OverallQual', 'PavedDrive', 'PoolArea', 'PoolQC',
       'RoofStyle', 'SaleCondition', 'SalePrice', 'SaleType', 'ScreenPorch',
       'Street', 'TotRmsAbvGrd', 'TotalBsmtSF', 'Utilities', 'WoodDeckSF',
       'TotalLivingSF', 'TotalMainBath',
       'TotalBath', 'AgeSold', 'TotalPorchSF', 'TotalSF', 'TotalArea',
       'GarageAgeSold', 'LastRemodelYrs', 'NhoodMedianPrice', 'NhoodMeanPrice',
       'NhoodMeanSF', 'NhoodMeanPricePerSF', 'ProxyPrice',  
       'FixerUpperScore', 'LotAreaCut', 'Shed', 'SeasonSold',
       'TotalHouseOverallQual', 'Functional_OverallQual', 'TotalSF_LotArea',
       'TotalSF_Condition']]

train = full_v6[:n_train]
test = full_v6[n_train:]

X_v6 = train.drop(['SalePrice'], axis=1)
test_X_v6 = test.drop(['SalePrice'], axis=1)
y_v6 = train['SalePrice']

X_scaled_v6 = scaler.fit(X_v6).transform(X_v6)
y_log_v6 = np.log(train['SalePrice'])
test_X_scaled_v6 = scaler.transform(test_X_v6)

lr = LinearRegression()
ridge = Ridge(alpha=40)

# Best parameters found:  {'alpha': 0.0001} 0.11187841593433792
lasso = Lasso(alpha=0.0001)

lasso_lars = LassoLarsIC()

# Best parameters found:  {'max_depth': 120, 'n_estimators': 800} 0.13093022119435033
rfr = RandomForestRegressor(max_depth=120, n_estimators=800)

# Best parameters found:  {'max_depth': 60, 'n_estimators': 600} 0.12379414612361438
etr = ExtraTreesRegressor(max_depth=60, max_features='auto', n_estimators=600)

gbr = GradientBoostingRegressor()

# Best parameters found:  {'colsample_bytree': 0.5, 'max_depth': 2, 'n_estimators': 600}
xgb = XGBRegressor(max_depth=2, n_estimators=600, colsample_bytree=0.5)

# Best parameters found:  {'C': 100, 'epsilon': 0.0, 'gamma': 0.001} 
svr = SVR(C=100, gamma=0.0001, epsilon=0.0)

# Best parameters found:  {'C': 0.5} 
linear_svr = LinearSVR(C=0.5)

# en = ElasticNet(alpha=0.004, l1_ratio=0.3, max_iter=10000)
en = ElasticNet(alpha=0.005, l1_ratio=0.1, max_iter=1000)

br = BayesianRidge()

# Best parameters found:  {'alpha': 1.0, 'coef0': 6.0, 'degree': 2, 'kernel': 'polynomial'} 0.11116062462422147
kr = KernelRidge(alpha=1.0, kernel='polynomial', degree=2, coef0=6.0)


regression_list = [lr, ridge, lasso, lasso_lars, rfr, etr, gbr, xgb, svr, linear_svr, en, br, kr]
name_list = ["Linear", "Ridge", "Lasso", "Lasso Lars", "Random Forest", "Extra Trees", "Grad Boost", "XGBoost", "SVR", "LinSVR", 
             "ElasticNet", "Bayesian Ridge", "Kernel Ridge"]
v6 = CvScore(regression_list, name_list, X_scaled_v6, y_log_v6)
v6.cv_list()
# even weighted model
var = 6

w1 = 1/var
w2 = 1/var
w3 = 1/var
w4 = 1/var
w5 = 1/var
w6 = 1/var

weight_avg_v6 = AverageWeight(mod = [en, lasso, ridge, xgb, svr, kr],weight=[w1,w2,w3,w4,w5,w6])

score = rmse_cv(weight_avg_v6,X_scaled_v6,y_log_v6)
print(score.mean())
a = Imputer().fit_transform(X_scaled_v6)
b = Imputer().fit_transform(y_log_v6.values.reshape(-1,1)).ravel()

final_model_v2 = AverageWeight(mod = [en, lasso, ridge, xgb, svr, kr],weight=[w1,w2,w3,w4,w5,w6])
final_model_v2 = final_model_v2.fit(a,b)
pred_v2 = np.exp(final_model_v2.predict(test_X_scaled_v6))
print(pred_v2)

id = test_orig['Id']
submission_v2 =pd.DataFrame({'Id': id, 'SalePrice':pred_v2})
submission_v2.to_csv("submission_v2.csv",index=False)