# Main Libraries

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from scipy.stats import norm, skew, kurtosis

import math

import os



# import operator for dictionary sorting operations

import operator



# preprocessing imports

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder



# train-test split

from sklearn.model_selection import train_test_split



# linear regression models

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor



# cross val, scored, scaler

from sklearn.model_selection import GridSearchCV,  cross_val_score

from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import RobustScaler



# boxcox normalisation

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



# stacking

from sklearn.base import BaseEstimator, RegressorMixin, clone



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

train_original = pd.read_csv('../input/train.csv')

test_original = pd.read_csv('../input/test.csv')
# create train and test datasets

train = train_original.copy()

test = test_original.copy()
# drop outliers suggested by dataset author

train = train.drop(train[train["GrLivArea"] > 4000].index, inplace=False)
# combine dataframes for simplicity

full = pd.concat([train,test], ignore_index=True)
full.head().transpose()
full.describe().transpose()
full.info()
print('Train Shape: ' + str(np.shape(train)) + '\n' + 'Test Shape: ' + str(np.shape(test)) + '\n' + 'Full Shape: ' + str(np.shape(full)))

categories = train.select_dtypes(include=['object']).columns

numericals = train.select_dtypes(include=['float64', 'int64']).columns

print('Num of Categories:  ' + str(len(categories)) + '\n' + 'Num of Values:  ', str(len(numericals)))
# Function provides details of missing columns

def missing_data_report(dataset, train_or_test = "Train"):

    missing_data = dataset.isnull().sum().sort_values(ascending=False)

    missing_data_percent = ((dataset.isnull().sum() / dataset.isnull().count()) * 100).sort_values(ascending=False)

    

    missing_report = pd.concat([missing_data, missing_data_percent], axis=1, keys=['Total', 'Percentage'])

    missing_report.rename_axis(train_or_test, inplace=True)

    

    return missing_report[missing_report > 0]



# Function to customise table display

def multi_table(table_list):

    from IPython.core.display import HTML

    return HTML(

        '<table><tr style="background-color:white;">' +

        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +

        '</tr></table>'

    )
# Dataframe of missing data

train_missing, test_missing = missing_data_report(train, "Train"), missing_data_report(train, "Test")



# Top Ten Missing data report

multi_table([train_missing.head(10), test_missing.head(10)])
# Check for NAs



def checknulls(df):

    nullcols = df.isnull().sum().sort_values(ascending=False)

    return nullcols[nullcols>0]



checknulls(full)
corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat,vmax=.8, square=True)
sig_numerical = ["SalePrice", "LotArea", "OverallQual", "OverallCond", "1stFlrSF", "2ndFlrSF", "BedroomAbvGr"]

dis = train[sig_numerical].hist(bins=15, figsize=(15,6), layout=(2,4))

sig_categorical = ["MSZoning", "LotShape", "Neighborhood", "CentralAir", "SaleCondition", "MoSold", "YrSold"]

fig, ax = plt.subplots(2,4, figsize=(20,10))



# Loop over every categorical variable to create countplot

for var, subplot in zip(sig_categorical, ax.flatten()):

    sns.countplot(train[var], ax=subplot)

    for label in subplot.get_xticklabels():

        label.set_rotation(90)
k = 10 #number of variables for heatmap

cormat=train.corr()

cols = cormat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
chartA = sns.pairplot(data=train,x_vars=['OverallQual', 'GrLivArea', 'ExterQual', 'KitchenQual',

       'GarageCars'],y_vars=['SalePrice'])



chartB = sns.pairplot(data=train,x_vars=['GarageArea', 'TotalBsmtSF', '1stFlrSF', 'BsmtQual'],y_vars=['SalePrice'])
sig_categorical = ["MSZoning", "LotShape", "Neighborhood", "CentralAir", "SaleCondition", "MoSold", "YrSold"]

fig, ax = plt.subplots(2,4, figsize=(20,10))



# Loop over every categorical variable to create countplot

for var, subplot in zip(sig_categorical, ax.flatten()):

    sns.countplot(train[var], ax=subplot)

    for label in subplot.get_xticklabels():

        label.set_rotation(90)
train['SalePrice'].describe()
sns.distplot(train['SalePrice'])

print('Kurt: ' + str(kurtosis(train['SalePrice'])) + '\n' + 'Skew: ' +str(skew(train['SalePrice'])))
plt.figure(1); plt.title('Normal')

sns.distplot(train['SalePrice'], kde=False, fit=stats.norm)



plt.figure(2); plt.title('Log Norm')

sns.distplot(train['SalePrice'], kde=False, fit=stats.lognorm)
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



    df['TotalHouseOverallQual'] = df['TotalSF'] * df['OverallQual']

    df['Functional_OverallQual'] = df['Functional'] * df['OverallQual']

    df['TotalSF_LotArea'] = df['TotalSF'] * df['LotArea']

    df['TotalSF_Condition'] = df['TotalSF'] * df['Condition1']



    return df
full = df_transform(full)
n_train=train.shape[0]

df = full[:n_train]
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in full.drop(['FxUp_SaleCond'], axis=1):

    if full[i].dtype in numeric_dtypes: 

        numerics2.append(i)



skew_features = full[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

skews = pd.DataFrame({'skew':skew_features})

skews = skews.drop(['SalePrice'], axis=0)
high_skew = skew_features[skew_features > 0.75]

skew_index = high_skew.index

   

for i in high_skew.index:

    full[i]= boxcox1p(full[i], boxcox_normmax(full[i]+1))

        

skew_features2 = full[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

skews2 = pd.DataFrame({'skew':skew_features2})

print(skews2.to_string())
allset = full[['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'Alley', 'BedroomAbvGr',

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



train = allset[:n_train]

test = allset[n_train:]



scaler = RobustScaler()



X_train = train.drop(['SalePrice'], axis=1)

X_test = test.drop(['SalePrice'], axis=1)

y_train = train['SalePrice']



X_train_scaled = scaler.fit(X_train).transform(X_train)

y_train_logged = np.log(train['SalePrice'])

X_test_scaled= scaler.transform(X_test)



X_train_scaled_df = pd.DataFrame(X_train_scaled)

y_logged_df = pd.DataFrame(y_train_logged)

X_test_scaled_df = pd.DataFrame(X_test_scaled)
lasso_fi=Lasso(alpha=0.001)

lasso_fi.fit(X_train_scaled,y_train_logged)

FI_lasso = pd.DataFrame({"Feature Importance":lasso_fi.coef_}, index=X_train.columns)

FI_sorted = FI_lasso.sort_values("Feature Importance",ascending=False)

print(FI_sorted.to_string())
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



# Best parameters found:  {'alpha': 40} 0.1120323653109581

ridge = Ridge(alpha=40)



# Best parameters found:  {'alpha': 0.0001} 0.11267514926665378

lasso = Lasso(alpha=0.0001)



# Best parameters found:  {'max_depth': 80, 'n_estimators': 800} 0.12673567602317132

rfr = RandomForestRegressor(max_depth=80, n_estimators=800)



regression_list = [lr, ridge, lasso, rfr]

name_list = ["Linear", "Ridge", "Lasso", "Random Forest"]
scores = CvScore(regression_list, name_list, X_train_scaled,y_train_logged)

scores.cv_list()
class grid():

    def __init__(self,model):

        self.model = model

    

    def grid_get(self,X,y,param_grid):

        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")

        grid_search.fit(X,y)

        print("Best parameters found: ", grid_search.best_params_, np.sqrt(-grid_search.best_score_))

        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])

        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
param_grid={'alpha':[0.00001, 0.0001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 4, 10, 20, 25, 30, 35, 40, 60, 100, 250, 500]}

grid(Ridge()).grid_get(X_train_scaled,y_train_logged,param_grid)
param_grid={'alpha':[0.00001, 0.0001, 0.01, 0.1, 0.25, 0.5, 0.75, 1]}

grid(Lasso()).grid_get(X_train_scaled,y_train_logged,param_grid)
# Random Forest Regression

rfr_param_grid = {

     'n_estimators':[100, 200, 400, 600, 800],

     'max_depth': [80, 120, 160, 200]

}



grid(RandomForestRegressor()).grid_get(X_train_scaled,y_train_logged,rfr_param_grid)
# define cross validation strategy

def rmse_cv(model,X,y):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))

    return rmse



# class object group for ensembling the model predictions together

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

variables = 4



w1 = 1/variables

w2 = 1/variables

w3 = 1/variables

w4 = 1/variables



weight_avg = AverageWeight(mod = [lr, lasso, ridge, rfr],weight=[w1,w2,w3,w4])



score = rmse_cv(weight_avg,X_train_scaled,y_train_logged)

print(score.mean())
a = Imputer().fit_transform(X_train_scaled)

b = Imputer().fit_transform(y_train_logged.values.reshape(-1,1)).ravel()



ensemble_model = AverageWeight(mod = [lr, lasso, ridge, rfr],weight=[w1,w2,w3,w4])

ensemble_model = ensemble_model.fit(a,b)
prediction = np.exp(ensemble_model.predict(X_test_scaled))

print(prediction)



id_list = test_original['Id']

submission =pd.DataFrame({'Id': id_list, 'SalePrice': prediction})

submission#.to_csv("submission.csv",index=False)