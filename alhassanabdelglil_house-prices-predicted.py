import pandas as pd
import numpy as np 
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
# train.head()
df_train = train.drop(['Id','Alley','Utilities','LotFrontage','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
# train.isnull().sum()
c_train = train.loc[:,['Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition']]
n_train = train.loc[:,['MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','BsmtFullBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageFinish','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MiscVal','MoSold','MoSold']]
# c_train

c_train = pd.DataFrame(c_train)
c_train.fillna(method='bfill',inplace=True)
n_train = pd.DataFrame(n_train)
n_train.fillna(method='bfill',inplace=True)

y_train = train.iloc[:,-1]
# y_train
c_train.isnull().sum()
# n_train.isnull().sum()
#catagorical data
c_train = pd.get_dummies(c_train,drop_first=True)
c_train
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
n_train = sc_X.fit_transform(n_train)
n_train

