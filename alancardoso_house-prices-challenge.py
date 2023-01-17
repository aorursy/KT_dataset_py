# Understand the problem. 
#We'll look at each variable and do a philosophical analysis about their meaning and importance for this problem.
# Univariable study. 
#We'll just focus on the dependent variable ('SalePrice') and try to know a little bit more about it.
# Multivariate study. 
#We'll try to understand how the dependent variable and independent variables relate.
# Basic cleaning. 
#We'll clean the dataset and handle the missing data, outliers and categorical variables.
# Test assumptions. 
#We'll check if our data meets the assumptions required by most multivariate techniques.
# Modelos: Linear Regression / Decision Tree Regressor / Random Forest Regression
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('dados/train.csv')
df_test = pd.read_csv('dados/test.csv')
df_train.info()
df_train['SalePrice'].describe()
df_train.columns
# Colunas importantes para o modelo:
# SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
# MSSubClass: The building class
# MSZoning: The general zoning classification
# LotFrontage: Linear feet of street connected to property
# LotArea: Lot size in square feet
# Street: Type of road access -> retirar do modelo
# Alley: Type of alley access -> retirar do modelo por muitos nulos
# LotShape: General shape of property -> retirar do modelo
# LandContour: Flatness of the property -> retirar do modelo
# Utilities: Type of utilities available -> retirar do modelo
# LotConfig: Lot configuration
# LandSlope: Slope of property
# Neighborhood: Physical locations within Ames city limits -> retirar do modelo
# Condition1: Proximity to main road or railroad -> retirar do modelo
# Condition2: Proximity to main road or railroad (if a second is present) -> retirar do modelo
# BldgType: Type of dwelling
# HouseStyle: Style of dwelling
# OverallQual: Overall material and finish quality ?
# OverallCond: Overall condition rating ?
# YearBuilt: Original construction date
# YearRemodAdd: Remodel date
# RoofStyle: Type of roof -> retirar do modelo
# RoofMatl: Roof material -> retirar do modelo
# Exterior1st: Exterior covering on house -> retirar do modelo
# Exterior2nd: Exterior covering on house (if more than one material) -> retirar do modelo
# MasVnrType: Masonry veneer type -> retirar do modelo
# MasVnrArea: Masonry veneer area in square feet -> retirar do modelo
# ExterQual: Exterior material quality
# ExterCond: Present condition of the material on the exterior -> retirar do modelo
# Foundation: Type of foundation
# BsmtQual: Height of the basement -> retirar do modelo
# BsmtCond: General condition of the basement -> retirar do modelo
# BsmtExposure: Walkout or garden level basement walls -> retirar do modelo
# BsmtFinType1: Quality of basement finished area
# BsmtFinSF1: Type 1 finished square feet -> retirar do modelo
# BsmtFinType2: Quality of second finished area (if present) -> retirar do modelo
# BsmtFinSF2: Type 2 finished square feet -> retirar do modelo
# BsmtUnfSF: Unfinished square feet of basement area -> retirar do modelo
# TotalBsmtSF: Total square feet of basement area -> retirar do modelo
# Heating: Type of heating -> retirar do modelo
# HeatingQC: Heating quality and condition -> retirar do modelo
# CentralAir: Central air conditioning - retirar do modelo
# Electrical: Electrical system -> retirar do modelo
# 1stFlrSF: First Floor square feet -> retirar do modelo
# 2ndFlrSF: Second floor square feet -> retirar do modelo
# LowQualFinSF: Low quality finished square feet (all floors) -> retirar do modelo
# GrLivArea: Above grade (ground) living area square feet -> retirar do modelo
# BsmtFullBath: Basement full bathrooms
# BsmtHalfBath: Basement half bathrooms -> retirar do modelo
# FullBath: Full bathrooms above grade
# HalfBath: Half baths above grade
# Bedroom: Number of bedrooms above basement level
# Kitchen: Number of kitchens -> retirar do modelo
# KitchenQual: Kitchen quality -> retirar do modelo
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms) -> retirar do modelo
# Functional: Home functionality rating -> retirar do modelo
# Fireplaces: Number of fireplaces -> retirar do modelo
# FireplaceQu: Fireplace quality -> retirar do modelo, muitos nulos
# GarageType: Garage location -> retirar do modelo
# GarageYrBlt: Year garage was built -> retirar do modelo
# GarageFinish: Interior finish of the garage -> retirar do modelo
# GarageCars: Size of garage in car capacity
# GarageArea: Size of garage in square feet -> retirar do modelo
# GarageQual: Garage quality -> retirar do modelo
# GarageCond: Garage condition -> retirar do modelo
# PavedDrive: Paved driveway -> retirar do modelo
# WoodDeckSF: Wood deck area in square feet -> retirar do modelo
# OpenPorchSF: Open porch area in square feet -> retirar do modelo
# EnclosedPorch: Enclosed porch area in square feet -> retirar do modelo
# 3SsnPorch: Three season porch area in square feet -> retirar do modelo
# ScreenPorch: Screen porch area in square feet -> retirar do modelo
# PoolArea: Pool area in square feet -> retirar do modelo
# PoolQC: Pool quality -> retirar do modelo
# Fence: Fence quality -> retirar do modelo
# MiscFeature: Miscellaneous feature not covered in other categories -> retirar do modelo
# MiscVal: $Value of miscellaneous feature -> retirar
# MoSold: Month Sold
# YrSold: Year Sold
# SaleType: Type of sale
# SaleCondition: Condition of sale
df_train.drop(['Street','Alley','LandContour','LandSlope','Utilities','Neighborhood','Condition1','Condition2','RoofStyle',
               'RoofMatl','MasVnrType','MasVnrArea','ExterCond', 'BsmtCond','BsmtExposure','BsmtFinSF1','BsmtFinType2',
               'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating', 'HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF',
               'LowQualFinSF','GrLivArea','BsmtHalfBath','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces',
               'FireplaceQu', 'GarageType','GarageYrBlt','GarageFinish','GarageArea','GarageQual','GarageCond','PavedDrive',
               'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolQC','Fence','MiscFeature', 'MiscVal',
               'Exterior1st','Exterior2nd','BsmtQual', 'PoolArea'
              ], axis=1, inplace=True)
df_train.head()
df_train = df_train[df_train.LotFrontage < df_train.LotFrontage.quantile(.99)]
sns.scatterplot(x='LotFrontage', y='SalePrice', data=df_train)
plt.title('LotFrontage x SalePrice')
plt.show()
df_train = df_train[df_train.LotArea < df_train.LotArea.quantile(.99)]
sns.scatterplot(x='LotArea', y='SalePrice', data=df_train)
plt.title('LotArea x SalePrice')
plt.show()
sns.distplot(df_train['SalePrice'])
df_train.BldgType.value_counts()
df_test.BldgType.value_counts()
df_train.HouseStyle.value_counts()
df_test.HouseStyle.value_counts()
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#df_train
dummies = pd.get_dummies(df_train.MSZoning)
df_train[['RL', 'RM', 'FV','RH', 'C']] = dummies
df_train.drop(['MSZoning'], axis=1, inplace=True)

#df_test
dummies = pd.get_dummies(df_test.MSZoning)
df_test[['RL', 'RM', 'FV','RH', 'C']] = dummies
#df_train
dummies = pd.get_dummies(df_train.LotShape)
df_train[['Reg', 'IR1', 'IR2','IR3']] = dummies
df_train.drop(['LotShape'], axis=1, inplace=True)

#df_test
dummies = pd.get_dummies(df_test.LotShape)
df_test[['Reg', 'IR1', 'IR2','IR3']] = dummies
#df_train
dummies = pd.get_dummies(df_train.LotConfig)
df_train[['Inside', 'Corner', 'CulDSac','FR2','FR3']] = dummies
df_train.drop(['LotConfig'], axis=1, inplace=True)

#df_test
dummies = pd.get_dummies(df_test.LotConfig)
df_test[['Inside', 'Corner', 'CulDSac','FR2','FR3']] = dummies
#df_train
dummies = pd.get_dummies(df_train.BldgType)
df_train[['1Fam', 'TwnhsE', 'Twnhs','Duplex','2fmCon']] = dummies
df_train.drop(['BldgType'], axis=1, inplace=True)

#df_test
dummies = pd.get_dummies(df_test.BldgType)
df_test[['1Fam', 'TwnhsE', 'Twnhs','Duplex','2fmCon']] = dummies
#df_train
dummies = pd.get_dummies(df_train.HouseStyle)
df_train[['1Story', '2Story', '1.5Fin','SLvl','SFoyer','1.5Unf','2.5Unf','2.5Fin']] = dummies
df_train.drop(['HouseStyle'], axis=1, inplace=True)

#df_test
dummies = pd.get_dummies(df_test.HouseStyle)
df_test[['1Story', '2Story', '1.5Fin','SLvl','SFoyer','1.5Unf','2.5Unf',]] = dummies
#df_train
dummies = pd.get_dummies(df_train.ExterQual)
df_train[['TA', 'Gd', 'Ex','Fa']] = dummies
df_train.drop(['ExterQual'], axis=1, inplace=True)

#df_test
dummies = pd.get_dummies(df_test.ExterQual)
df_test[['TA', 'Gd', 'Ex','Fa']] = dummies
#df_train
dummies = pd.get_dummies(df_train.Foundation)
df_train[['PConc', 'CBlock', 'BrkTil','Slab','Stone','Wood']] = dummies
df_train.drop(['Foundation'], axis=1, inplace=True)

#df_test
dummies = pd.get_dummies(df_test.Foundation)
df_test[['PConc', 'CBlock', 'BrkTil','Slab','Stone','Wood']] = dummies
#df_train
dummies = pd.get_dummies(df_train.BsmtFinType1)
df_train[['Unf', 'GLQ', 'ALQ','BLQ','Rec','LwQ']] = dummies
df_train.drop(['BsmtFinType1'], axis=1, inplace=True)

#df_test
dummies = pd.get_dummies(df_test.BsmtFinType1)
df_test[['Unf', 'GLQ', 'ALQ','BLQ','Rec','LwQ']] = dummies
#df_train
dummies = pd.get_dummies(df_train.SaleType)
df_train[['WD', 'New', 'COD','ConLD','ConLw','CWD','ConLI','Oth','Con']] = dummies
df_train.drop(['SaleType'], axis=1, inplace=True)

#df_test
dummies = pd.get_dummies(df_test.SaleType)
df_test[['WD', 'New', 'COD','ConLD','ConLw','CWD','ConLI','Oth','Con']] = dummies
#df_traim
dummies = pd.get_dummies(df_train.SaleCondition)
df_train[['Normal', 'Partial', 'Abnorml','Family','Alloca','AdjLand']] = dummies
df_train.drop(['SaleCondition'], axis=1, inplace=True)

#df_test
dummies = pd.get_dummies(df_test.SaleCondition)
df_test[['Normal', 'Partial', 'Abnorml','Family','Alloca','AdjLand']] = dummies
df_train.head(10)
#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GarageCars', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();
#df_train['SalePrice_log'] = np.log(df_train.SalePrice)
df_train.head()
from sklearn.model_selection import train_test_split

features = df_train.columns.difference(['Id','SalePrice'])

x = df_train[features]
y = df_train['SalePrice']
#y = df_train['SalePrice_log']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.get_params()
rf.fit(x_train, y_train) # treinando
pred = rf.predict(x_test) # prevendo
from sklearn.metrics import r2_score, mean_squared_error
print(f'R2 modelo2: {r2_score(y_test, pred)}') # R2 Score
print(f'MSE modelo2: {mean_squared_error(y_test, pred)}') # MSE
sns.scatterplot(df_train.LotArea, df_train.SalePrice)
plt.title('Área coberta X Preço')
plt.show()
features_teste = df_test.columns.difference(['MSZoning','LotShape','LotConfig','BldgType','HouseStyle','Street','Alley',
               'LandContour','LandSlope','Utilities','Neighborhood','Condition1','Condition2','RoofStyle','PoolArea','ExterQual',
               'RoofMatl','MasVnrType','MasVnrArea','ExterCond', 'BsmtCond','BsmtExposure','BsmtFinSF1','BsmtFinType2',
               'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating', 'HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF',
               'LowQualFinSF','GrLivArea','BsmtHalfBath','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces',
               'FireplaceQu', 'GarageType','GarageYrBlt','GarageFinish','GarageArea','GarageQual','GarageCond','PavedDrive',
               'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolQC','Fence','MiscFeature', 'MiscVal',
               'Exterior1st','Exterior2nd','BsmtQual','Exterqual','Foundation','BsmtFinType1','SaleType','SaleCondition' ])
features_teste
df_test['BsmtFullBath'] = df_test['BsmtFullBath'].transform(lambda x: x.fillna(x.median()))
df_test['LotFrontage'] = df_test['LotFrontage'].transform(lambda x: x.fillna(x.median()))
df_test['GarageCars'] = df_test['GarageCars'].transform(lambda x: x.fillna(x.median()))
teste = df_test[features_teste]
previsao = rf.predict(teste) # prevendo valores
#previsao_log = rf.predict(teste) # prevendo valores
#previsao_log
#preco = np.exp(previsao_log)
#preco
gender_submission = pd.DataFrame({'Id': df_test.Id, 
                                  'SalePrice': previsao})
gender_submission.head()
gender_submission.to_csv('dados/gender_submission.csv', index=False)
arquivo_gerado = pd.read_csv('dados/gender_submission.csv')
arquivo_gerado.head()
arquivo_gerado.tail()
