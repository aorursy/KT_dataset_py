# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df.head()
y=df['SalePrice']

df.drop(['SalePrice', 'Id'], axis=1, inplace=True)
df.shape
for i in df.columns:

    print(i, df[i].isnull().sum())
df.drop(['MiscFeature', 'Fence' , 'PoolQC', 'Alley'], axis=1, inplace=True)
df.head()
for i in df.columns:

    print(i, (df[i].isnull().sum()/1460)*100)
df['GarageCond'].isnull().sum()
df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)
df['MasVnrType'].fillna('NA', inplace=True)
df['MasVnrArea'].fillna(df['MasVnrArea'].mean(), inplace=True)
df['BsmtQual'].fillna('NA', inplace=True)
df['BsmtCond'].fillna('NA', inplace=True)
df['BsmtExposure'].fillna('NA', inplace=True)
df['BsmtFinType1'].fillna('NA', inplace=True)
df['BsmtFinType2'].fillna('NA', inplace=True)
df['Electrical'].fillna(method='ffill', inplace=True)
df['FireplaceQu'].fillna('NA', inplace=True)
df['GarageType'].fillna('NA', inplace=True)
df['GarageYrBlt'].fillna(method='ffill', inplace=True)
df['GarageFinish'].fillna('NA', inplace=True)
df['GarageQual'].fillna('NA', inplace=True)
df['GarageCond'].fillna('NA', inplace=True)
df.isnull().sum().sum()
df.head(10)
with open('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt', 'r') as f:

    data=f.read()

    print(data)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

mms=MinMaxScaler()
df['LowQualFinSF']=mms.fit_transform(df.loc[:, ['LowQualFinSF']])
df['BsmtFinSF2']=mms.fit_transform(df.loc[:, ['BsmtFinSF2']])
df['WoodDeckSF']=mms.fit_transform(df.loc[:, ['WoodDeckSF']])
df['OpenPorchSF']=mms.fit_transform(df.loc[:, ['OpenPorchSF']])
df['3SsnPorch']=mms.fit_transform(df.loc[:, ['3SsnPorch']])
df['ScreenPorch']=mms.fit_transform(df.loc[:, ['ScreenPorch']])
df['PoolArea']=mms.fit_transform(df.loc[:, ['PoolArea']])
df['MiscVal']=mms.fit_transform(df.loc[:, ['MiscVal']])
df['MSSubClass']=mms.fit_transform(df.loc[:, ['MSSubClass']])
df['YearBuilt']=mms.fit_transform(df.loc[:, ['YearBuilt']])
df['YearRemodAdd']=mms.fit_transform(df.loc[:, ['YearRemodAdd']])
df['GarageYrBlt']=mms.fit_transform(df.loc[:, ['GarageYrBlt']])
df.head()
from sklearn.preprocessing import StandardScaler

sts=StandardScaler()
df['LotFrontage']=sts.fit_transform(df.loc[:, ['LotFrontage']])
df['LotArea']=sts.fit_transform(df.loc[:, ['LotArea']])
df['MasVnrArea']=sts.fit_transform(df.loc[:, ['MasVnrArea']])
df['BsmtFinSF1']=sts.fit_transform(df.loc[:, ['BsmtFinSF1']])
df['BsmtUnfSF']=sts.fit_transform(df.loc[:, ['BsmtUnfSF']])
df['TotalBsmtSF']=sts.fit_transform(df.loc[:, ['TotalBsmtSF']])
df['1stFlrSF']=sts.fit_transform(df.loc[:, ['1stFlrSF']])
df['2ndFlrSF']=sts.fit_transform(df.loc[:, ['2ndFlrSF']])
df['GrLivArea']=sts.fit_transform(df.loc[:, ['GrLivArea']])
df['GarageArea']=sts.fit_transform(df.loc[:, ['GarageArea']])
df['EnclosedPorch']=sts.fit_transform(df.loc[:, ['EnclosedPorch']])
df['YrSold']=sts.fit_transform(df.loc[:, ['YrSold']])
df.head()
MSZoning_dummy=pd.get_dummies(df['MSZoning'])
MSZoning_dummy.columns
MSZoning_dummy.drop(['C (all)'], axis=1, inplace=True)
Street_dummy=pd.get_dummies(df['Street'])
Street_dummy.columns
Street_dummy.drop(['Grvl'], axis=1, inplace=True)
LotShape_dummy=pd.get_dummies(df['LotShape'])
LotShape_dummy.columns
LotShape_dummy.drop(['IR1'], axis=1, inplace=True)
LandContour_dummy=pd.get_dummies(df['LandContour'])
LandContour_dummy.columns
LandContour_dummy.drop(['Bnk'], axis=1, inplace=True)
Utilities_dummy=pd.get_dummies(df['Utilities'])
Utilities_dummy.columns
Utilities_dummy.drop(['AllPub'], axis=1, inplace=True)
LotConfig_dummy=pd.get_dummies(df['LotConfig'])
LotConfig_dummy.columns
LotConfig_dummy.drop(['Corner'], axis=1, inplace=True)
LandSlope_dummy=pd.get_dummies(df['LandSlope'])
LandSlope_dummy.columns
LandSlope_dummy.drop(['Gtl'], axis=1, inplace=True)
Neighborhood_dummy=pd.get_dummies(df['Neighborhood'])
Neighborhood_dummy.columns
Neighborhood_dummy.drop(['Blmngtn'], axis=1, inplace=True)
Condition1_dummy=pd.get_dummies(df['Condition1'])
Condition1_dummy.columns
Condition1_dummy.drop(['Artery'], axis=1, inplace=True)
Condition2_dummy=pd.get_dummies(df['Condition1'])
Condition2_dummy.columns
Condition2_dummy.drop(['Artery'], axis=1, inplace=True)
BldgType_dummy=pd.get_dummies(df['BldgType'])
BldgType_dummy.columns
BldgType_dummy.drop(['1Fam'], axis=1, inplace=True)
HouseStyle_dummy=pd.get_dummies(df['HouseStyle'])
HouseStyle_dummy.columns
HouseStyle_dummy.drop(['1.5Fin'], axis=1, inplace=True)
RoofStyle_dummy=pd.get_dummies(df['RoofStyle'])
RoofStyle_dummy.columns
RoofStyle_dummy.drop(['Flat'], axis=1, inplace=True)
RoofMatl_dummy=pd.get_dummies(df['RoofMatl'])
RoofMatl_dummy.columns
RoofMatl_dummy.drop(['ClyTile'], axis=1, inplace=True)
Exterior1st_dummy=pd.get_dummies(df['Exterior1st'])
Exterior1st_dummy.columns
Exterior1st_dummy.drop(['AsbShng'], axis=1, inplace=True)
Exterior2nd_dummy=pd.get_dummies(df['Exterior2nd'])
Exterior2nd_dummy.columns
Exterior2nd_dummy.drop(['AsbShng'], axis=1, inplace=True)
MasVnrType_dummy=pd.get_dummies(df['MasVnrType'])
MasVnrType_dummy.columns
MasVnrType_dummy.drop(['BrkCmn', 'NA'], axis=1, inplace=True)
ExterQual_dummy=pd.get_dummies(df['ExterQual'])
ExterQual_dummy.columns
ExterQual_dummy.drop(['Ex'], axis=1, inplace=True)
ExterCond_dummy=pd.get_dummies(df['ExterCond'])
ExterCond_dummy.columns
ExterCond_dummy.drop(['Ex'], axis=1, inplace=True)
Foundation_dummy=pd.get_dummies(df['Foundation'])
Foundation_dummy.columns
Foundation_dummy.drop(['BrkTil'], axis=1, inplace=True)
BsmtQual_dummy=pd.get_dummies(df['BsmtQual'])
BsmtQual_dummy.columns
BsmtQual_dummy.drop(['Ex', 'NA'], axis=1, inplace=True)
BsmtCond_dummy=pd.get_dummies(df['BsmtCond'])
BsmtCond_dummy.columns
BsmtCond_dummy.drop(['Fa', 'NA'], axis=1, inplace=True)
BsmtExposure_dummy=pd.get_dummies(df['BsmtExposure'])
BsmtExposure_dummy.columns
BsmtExposure_dummy.drop(['Av', 'NA'], axis=1, inplace=True)
BsmtFinType1_dummy=pd.get_dummies(df['BsmtFinType1'])
BsmtFinType1_dummy.columns
BsmtFinType1_dummy.drop(['ALQ', "NA"], axis=1, inplace=True)
BsmtFinType2_dummy=pd.get_dummies(df['BsmtFinType2'])
BsmtFinType2_dummy.columns
BsmtFinType2_dummy.drop(['ALQ', 'NA'], axis=1, inplace=True)
Heating_dummy=pd.get_dummies(df['Heating'])
Heating_dummy.columns
Heating_dummy.drop(['Floor'], axis=1, inplace=True)
HeatingQC_dummy=pd.get_dummies(df['HeatingQC'])
HeatingQC_dummy.columns
HeatingQC_dummy.drop(['Ex'], axis=1, inplace=True)
CentralAir_dummy=pd.get_dummies(df['CentralAir'])
CentralAir_dummy.columns
CentralAir_dummy.drop(['N'], axis=1 ,inplace=True)
Electrical_dummy=pd.get_dummies(df['Electrical'])
Electrical_dummy.columns
Electrical_dummy.drop('FuseA', axis=1, inplace=True)
KitchenQual_dummy=pd.get_dummies(df['KitchenQual'])
KitchenQual_dummy.columns
KitchenQual_dummy.drop(['Ex'], axis=1, inplace=True)
Functional_dummy=pd.get_dummies(df['Functional'])
Functional_dummy.columns
Functional_dummy.drop(['Maj1'], axis=1, inplace=True)
FireplaceQu_dummy=pd.get_dummies(df['FireplaceQu'])
FireplaceQu_dummy.columns
FireplaceQu_dummy.drop(['Ex', 'NA'], axis=1, inplace=True)
GarageType_dummy=pd.get_dummies(df['GarageType'])
GarageType_dummy.columns
GarageType_dummy.drop(['2Types', 'NA'], axis=1, inplace=True)
GarageFinish_dummy=pd.get_dummies(df['GarageFinish'])
GarageFinish_dummy.columns
GarageFinish_dummy.drop(['Fin', "NA"], axis=1, inplace=True)
GarageQual_dummy=pd.get_dummies(df['GarageQual'])
GarageQual_dummy.columns
GarageQual_dummy.drop(['Ex', 'NA'], axis=1, inplace=True)
GarageCond_dummy=pd.get_dummies(df['GarageCond'])
GarageCond_dummy.columns
GarageCond_dummy.drop(['Ex', 'NA'], axis=1, inplace=True)
PavedDrive_dummy=pd.get_dummies(df['PavedDrive'])
PavedDrive_dummy.columns
PavedDrive_dummy.drop(['N'], axis=1, inplace=True)
SaleType_dummy=pd.get_dummies(df['SaleType'])
SaleType_dummy.columns
SaleType_dummy.drop(['COD'], axis=1, inplace=True)
SaleCondition_dummy=pd.get_dummies(df['SaleCondition'])
SaleCondition_dummy.columns
SaleCondition_dummy.drop(['Abnorml'], axis=1, inplace=True)
for i in df.columns:

    print(i, df[i].unique())
df.drop(['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'], axis=1, inplace=True)
dummy_datasets=[df, MSZoning_dummy, Street_dummy, LotShape_dummy, LandContour_dummy, Utilities_dummy, LotConfig_dummy, LandSlope_dummy, Neighborhood_dummy, Condition1_dummy, Condition2_dummy, BldgType_dummy, HouseStyle_dummy, RoofStyle_dummy, RoofMatl_dummy, Exterior1st_dummy, Exterior2nd_dummy, MasVnrType_dummy, ExterQual_dummy, ExterCond_dummy, Foundation_dummy, BsmtQual_dummy, BsmtCond_dummy, BsmtExposure_dummy, BsmtFinType1_dummy, BsmtFinType2_dummy, Heating_dummy, HeatingQC_dummy, CentralAir_dummy, Electrical_dummy, KitchenQual_dummy, Functional_dummy, FireplaceQu_dummy, GarageType_dummy, GarageFinish_dummy, GarageQual_dummy,GarageCond_dummy, PavedDrive_dummy, SaleType_dummy, SaleCondition_dummy]
x=pd.concat(dummy_datasets, axis=1, sort=False)
x.head()
x.shape
x.isnull().sum().sum()
#from sklearn import decomposition



#pca = decomposition.PCA(n_components=176)

#pca.fit(x)

#X = pca.transform(x)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.2)
X_train.columns.unique()
from sklearn.model_selection import GridSearchCV

from sklearn import ensemble

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb



params = {'n_estimators': 20000, 'max_depth': 5, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls', 'max_features': 237, 'random_state': 0}

clf = ensemble.GradientBoostingRegressor(**params)

#clf= ensemble.AdaBoostRegressor(DecisionTreeRegressor(max_depth=7, max_features=237, random_state=0),

  #                        n_estimators=1500, random_state=0, learning_rate=1.0)

#clf =  SVR(kernel='rbf', gamma='scale', epsilon=0.01, C=1000000)

#clf= ensemble.RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=0, max_features=237)



#clf= model = xgb.XGBRegressor(colsample_bytree=0.8, gamma=0, learning_rate=0.01, max_depth=4,

 #                min_child_weight=2.0, n_estimators=100000, reg_alpha=0.75, reg_lambda=0.45, subsample=1)



clf.fit(X_train, y_train)
pred=clf.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2_score(y_test, pred)
mean_squared_error(y_test, pred)
mean_absolute_error(y_test, pred)
y_test[:10]
pred[:10]
test_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_df.head()
test_df.shape
test_df.columns
for i in test_df.columns:

    print(i, test_df[i].isnull().sum())
test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean(), inplace=True)
test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mean(), inplace=True)
test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean(), inplace=True)
test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mean(), inplace=True)
test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean(), inplace=True)
test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean(), inplace=True)
test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mean(), inplace=True)
test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mean(), inplace=True)
test_df['GarageYrBlt'].fillna(test_df['GarageYrBlt'].mean(), inplace=True)
test_df['GarageCars'].fillna(test_df['GarageCars'].mean(), inplace=True)
test_df['GarageArea'].fillna(test_df['GarageArea'].mean(), inplace=True)
def clean_data(df):

    df.drop(['Id', 'MiscFeature', 'Fence', 'PoolQC', 'Alley'], axis=1, inplace=True)

    

    df['LowQualFinSF']=mms.transform(df.loc[:, ['LowQualFinSF']])

    df['BsmtFinSF2']=mms.transform(df.loc[:, ['BsmtFinSF2']])

    df['WoodDeckSF']=mms.transform(df.loc[:, ['WoodDeckSF']])

    df['OpenPorchSF']=mms.transform(df.loc[:, ['OpenPorchSF']])

    df['3SsnPorch']=mms.transform(df.loc[:, ['3SsnPorch']])

    df['ScreenPorch']=mms.transform(df.loc[:, ['ScreenPorch']])

    df['PoolArea']=mms.transform(df.loc[:, ['PoolArea']])

    df['MiscVal']=mms.transform(df.loc[:, ['MiscVal']])

    df['MSSubClass']=mms.transform(df.loc[:, ['MSSubClass']])

    df['YearBuilt']=mms.transform(df.loc[:, ['YearBuilt']])

    df['YearRemodAdd']=mms.transform(df.loc[:, ['YearRemodAdd']])

    df['GarageYrBlt']=mms.transform(df.loc[:, ['GarageYrBlt']])

    

    df['LotFrontage']=sts.transform(df.loc[:, ['LotFrontage']])

    df['LotArea']=sts.transform(df.loc[:, ['LotArea']])

    df['MasVnrArea']=sts.transform(df.loc[:, ['MasVnrArea']])

    df['BsmtFinSF1']=sts.transform(df.loc[:, ['BsmtFinSF1']])

    df['BsmtUnfSF']=sts.transform(df.loc[:, ['BsmtUnfSF']])

    df['TotalBsmtSF']=sts.transform(df.loc[:, ['TotalBsmtSF']])

    df['1stFlrSF']=sts.transform(df.loc[:, ['1stFlrSF']])

    df['2ndFlrSF']=sts.transform(df.loc[:, ['2ndFlrSF']])

    df['GrLivArea']=sts.transform(df.loc[:, ['GrLivArea']])

    df['GarageArea']=sts.transform(df.loc[:, ['GarageArea']])

    df['EnclosedPorch']=sts.transform(df.loc[:, ['EnclosedPorch']])

    df['YrSold']=sts.transform(df.loc[:, ['YrSold']])

    

    MSZoning_dummy=pd.get_dummies(df['MSZoning'])

    MSZoning_dummy.drop(['C (all)'], axis=1, inplace=True)

    Street_dummy=pd.get_dummies(df['Street'])

    Street_dummy.drop(['Grvl'], axis=1, inplace=True)

    LotShape_dummy=pd.get_dummies(df['LotShape'])

    LotShape_dummy.drop(['IR1'], axis=1, inplace=True)

    LandContour_dummy=pd.get_dummies(df['LandContour'])

    LandContour_dummy.drop(['Bnk'], axis=1, inplace=True)

    Utilities_dummy=pd.get_dummies(df['Utilities'])

    Utilities_dummy.drop(['AllPub'], axis=1, inplace=True)

    LotConfig_dummy=pd.get_dummies(df['LotConfig'])

    LotConfig_dummy.drop(['Corner'], axis=1, inplace=True)

    LandSlope_dummy=pd.get_dummies(df['LandSlope'])

    LandSlope_dummy.drop(['Gtl'], axis=1, inplace=True)

    Neighborhood_dummy=pd.get_dummies(df['Neighborhood'])

    Neighborhood_dummy.drop(['Blmngtn'], axis=1, inplace=True)

    Condition1_dummy=pd.get_dummies(df['Condition1'])

    Condition1_dummy.drop(['Artery'], axis=1, inplace=True)

    Condition2_dummy=pd.get_dummies(df['Condition1'])

    Condition2_dummy.drop(['Artery'], axis=1, inplace=True)

    BldgType_dummy=pd.get_dummies(df['BldgType'])

    BldgType_dummy.drop(['1Fam'], axis=1, inplace=True)

    HouseStyle_dummy=pd.get_dummies(df['HouseStyle'])

    HouseStyle_dummy.drop(['1.5Fin'], axis=1, inplace=True)

    RoofStyle_dummy=pd.get_dummies(df['RoofStyle'])

    RoofStyle_dummy.drop(['Flat'], axis=1, inplace=True)

    RoofMatl_dummy=pd.get_dummies(df['RoofMatl'])

    RoofMatl_dummy.drop(['CompShg'], axis=1, inplace=True)

    Exterior1st_dummy=pd.get_dummies(df['Exterior1st'])

    Exterior1st_dummy.drop(['AsbShng'], axis=1, inplace=True)

    Exterior2nd_dummy=pd.get_dummies(df['Exterior2nd'])

    Exterior2nd_dummy.drop(['AsbShng'], axis=1, inplace=True)

    MasVnrType_dummy=pd.get_dummies(df['MasVnrType'])

    MasVnrType_dummy.drop(['BrkCmn'], axis=1, inplace=True)

    ExterQual_dummy=pd.get_dummies(df['ExterQual'])

    ExterQual_dummy.drop(['Ex'], axis=1, inplace=True)

    ExterCond_dummy=pd.get_dummies(df['ExterCond'])

    ExterCond_dummy.drop(['Ex'], axis=1, inplace=True)

    Foundation_dummy=pd.get_dummies(df['Foundation'])

    Foundation_dummy.drop(['BrkTil'], axis=1, inplace=True)

    BsmtQual_dummy=pd.get_dummies(df['BsmtQual'])

    BsmtQual_dummy.drop(['Ex'], axis=1, inplace=True)

    BsmtCond_dummy=pd.get_dummies(df['BsmtCond'])

    BsmtCond_dummy.drop(['Fa'], axis=1, inplace=True)

    BsmtExposure_dummy=pd.get_dummies(df['BsmtExposure'])

    BsmtExposure_dummy.drop(['Av'], axis=1, inplace=True)

    BsmtFinType1_dummy=pd.get_dummies(df['BsmtFinType1'])

    BsmtFinType1_dummy.drop(['ALQ'], axis=1, inplace=True)

    BsmtFinType2_dummy=pd.get_dummies(df['BsmtFinType2'])

    BsmtFinType2_dummy.drop(['ALQ'], axis=1, inplace=True)

    Heating_dummy=pd.get_dummies(df['Heating'])

    Heating_dummy.drop(['GasA'], axis=1, inplace=True)

    HeatingQC_dummy=pd.get_dummies(df['HeatingQC'])

    HeatingQC_dummy.drop(['Ex'], axis=1, inplace=True)

    CentralAir_dummy=pd.get_dummies(df['CentralAir'])

    CentralAir_dummy.drop(['N'], axis=1 ,inplace=True)

    Electrical_dummy=pd.get_dummies(df['Electrical'])

    Electrical_dummy.drop('FuseA', axis=1, inplace=True)

    KitchenQual_dummy=pd.get_dummies(df['KitchenQual'])

    KitchenQual_dummy.drop(['Ex'], axis=1, inplace=True)

    Functional_dummy=pd.get_dummies(df['Functional'])

    Functional_dummy.drop(['Maj1'], axis=1, inplace=True)

    FireplaceQu_dummy=pd.get_dummies(df['FireplaceQu'])

    FireplaceQu_dummy.drop(['Ex'], axis=1, inplace=True)

    GarageType_dummy=pd.get_dummies(df['GarageType'])

    GarageType_dummy.drop(['2Types'], axis=1, inplace=True)

    GarageFinish_dummy=pd.get_dummies(df['GarageFinish'])

    GarageFinish_dummy.drop(['Fin'], axis=1, inplace=True)

    GarageQual_dummy=pd.get_dummies(df['GarageQual'])

    GarageQual_dummy.drop(['Fa'], axis=1, inplace=True)

    GarageCond_dummy=pd.get_dummies(df['GarageCond'])

    GarageCond_dummy.drop(['Ex'], axis=1, inplace=True)

    PavedDrive_dummy=pd.get_dummies(df['PavedDrive'])

    PavedDrive_dummy.drop(['N'], axis=1, inplace=True)

    SaleType_dummy=pd.get_dummies(df['SaleType'])

    SaleType_dummy.drop(['COD'], axis=1, inplace=True)

    SaleCondition_dummy=pd.get_dummies(df['SaleCondition'])

    SaleCondition_dummy.drop(['Abnorml'], axis=1, inplace=True)

    

    df.drop(['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'], axis=1, inplace=True)

    dummy_datasets=[df, MSZoning_dummy, Street_dummy, LotShape_dummy, LandContour_dummy, Utilities_dummy, LotConfig_dummy, LandSlope_dummy, Neighborhood_dummy, Condition1_dummy, Condition2_dummy, BldgType_dummy, HouseStyle_dummy, RoofStyle_dummy, RoofMatl_dummy, Exterior1st_dummy, Exterior2nd_dummy, MasVnrType_dummy, ExterQual_dummy, ExterCond_dummy, Foundation_dummy, BsmtQual_dummy, BsmtCond_dummy, BsmtExposure_dummy, BsmtFinType1_dummy, BsmtFinType2_dummy, Heating_dummy, HeatingQC_dummy, CentralAir_dummy, Electrical_dummy, KitchenQual_dummy, Functional_dummy, FireplaceQu_dummy, GarageType_dummy, GarageFinish_dummy, GarageQual_dummy,GarageCond_dummy, PavedDrive_dummy, SaleType_dummy, SaleCondition_dummy]

    x=pd.concat(dummy_datasets, axis=1, sort=False)

    

    return x
processed_test_data=clean_data(test_df)
processed_test_data.shape
for i in range(225,238):

    processed_test_data[str(i)]=0
processed_test_data.shape
#from sklearn import decomposition



#pca = decomposition.PCA(n_components=176)

#pca.fit(processed_test_data)

#test_X = pca.transform(processed_test_data)
pred_test=clf.predict(processed_test_data)
pred_test
sample_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

sample_df.head()
test_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_df.head()
pred_test_df=pd.DataFrame(pred_test, columns=['SalePrice'])



submission_data=pd.concat([test_df['Id'], pred_test_df['SalePrice']], axis=1)
submission_data.to_csv(('submission_file.csv'), index=None, header=True)
from IPython.display import FileLink

FileLink(r'submission_file.csv')