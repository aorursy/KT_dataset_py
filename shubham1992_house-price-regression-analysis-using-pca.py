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
import pandas as pd

import numpy as np

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train.head()
train.shape
test.shape
#check for the columns of dataset

train.columns
a = train.isna().sum()
df = pd.DataFrame(a,columns=['value'])

df.head()
df[df['value']>750]
train = train.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)

test = test.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
df[df['value']==0].count()
#checking columns between 100 and 750 null values

df[(df['value']<750) & (df['value']>100)]
train['LotFrontage'].dtypes
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train['FireplaceQu'].dtypes
train['FireplaceQu'].unique()
train = train.replace({'TA':1,'Gd':2,'Fa':3,'Ex':4,'Po':5})
#replace all the null values with mode

train['FireplaceQu'].mode()
train['FireplaceQu'] = train['FireplaceQu'].fillna(2.0)
df[(df['value']<=100) & (df['value']>50)]
train['GarageType'].unique()
train = train.replace({'Attchd':1, 'Detchd':2, 'BuiltIn':3, 'CarPort':4, 'Basment':5, '2Types':6})
train['GarageType'].mode()
train['GarageType'] = train['GarageType'].fillna(1)
train['GarageFinish'].unique()
train = train.replace({'RFn':1, 'Unf':2, 'Fin':3})
train['GarageFinish'].mode()
train['GarageFinish'] = train['GarageFinish'].fillna(2.0)
train['GarageYrBlt'].dtypes
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())
train['GarageQual'].unique()
train['GarageCond'].unique()
train['GarageQual'].mode()
train['GarageQual'] = train['GarageQual'].fillna(1.0)

train['GarageCond'].mode()
train['GarageCond'] = train['GarageCond'].fillna(1.0)
df[(df['value']<=50) & (df['value']>25)]
train['BsmtQual'].mode()
train['BsmtQual'] = train['BsmtQual'].fillna(1)
train['BsmtCond'].mode()
train['BsmtCond'] = train['BsmtCond'].fillna(1)
train['BsmtExposure'].unique()
train = train.replace({'No':1 ,'Mn':3, 'Av':4})
train['BsmtExposure'].mode()
train['BsmtExposure'] = train['BsmtExposure'].fillna(1)
train['BsmtFinType1'].unique()
train = train.replace({'GLQ':1, 'ALQ':2, 'Unf':3, 'Rec':4, 'BLQ':5, 'LwQ':6})
train['BsmtFinType1'].mode()
train['BsmtFinType1'] = train['BsmtFinType1'].fillna(2)
train['BsmtFinType2'].mode()
train['BsmtFinType2'] = train['BsmtFinType2'].fillna(2.0)
df[(df['value']<=25) & (df['value']>0)]
train['MasVnrType'].unique()
train = train.replace({'BrkFace':1, 'None':2, 'Stone':3, 'BrkCmn':4})
train['MasVnrType'].mode()
train['MasVnrType'] = train['MasVnrType'].fillna(2)
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
train['Electrical'].unique()
train = train.replace({'SBrkr':1, 'FuseF':2, 'FuseA':3, 'FuseP':4, 'Mix':5})
train['Electrical'].mode()
train['Electrical'] = train['Electrical'].fillna(1)
train.select_dtypes(include ='object').columns
pd.concat([train['MSZoning'],train['Street'],train['LotShape'],train['LandContour'],train['Utilities'],train['LotConfig'],

           train['LandSlope'],train['Neighborhood'],train['Condition1'],train['Condition2'],train['BldgType'],

           train['HouseStyle'],train['RoofStyle'],train['RoofMatl'],train['Exterior1st'],train['Exterior2nd'],

           train['Foundation'],train['Heating'],train['CentralAir'],train['Functional'],train['GarageFinish'],

           train['PavedDrive'],train['SaleType'],train['SaleCondition']]).unique()
train = train.replace({'RL':1, 'RM':2, 'C (all)':3, 'FV':4, 'RH':5, 'Pave':6, 'Grvl':7, 'Reg':8, 'IR1':9,'IR2':10, 'IR3':11,

                       'Lvl':12, 'Bnk':13, 'Low':14, 'HLS':15, 'AllPub':16, 'NoSeWa':17,'Inside':18, 'FR2':19, 'Corner':20,

                       'CulDSac':21, 'FR3':22, 'Gtl':23, 'Mod':24, 'Sev':25,'CollgCr':26, 'Veenker':27, 'Crawfor':28,

                       'NoRidge':29, 'Mitchel':30, 'Somerst':31,'NWAmes':32, 'OldTown':33, 'BrkSide':34, 'Sawyer':35,

                       'NridgHt':36, 'NAmes':37,'SawyerW':38, 'IDOTRR':39, 'MeadowV':40, 'Edwards':41, 'Timber':42,

                       'Gilbert':43,'StoneBr':44, 'ClearCr':45, 'NPkVill':46, 'Blmngtn':47, 'BrDale':48, 'SWISU':49,

                       'Blueste':50, 'Norm':51, 'Feedr':52, 'PosN':53, 'Artery':54, 'RRAe':55, 'RRNn':56,'RRAn':57,

                       'PosA':58, 'RRNe':59, '1Fam':60, '2fmCon':61, 'Duplex':62, 'TwnhsE':63,'Twnhs':64, '2Story':65,

                       '1Story':66, '1.5Fin':67, '1.5Unf':68, 'SFoyer':69, 'SLvl':70,'2.5Unf':71, '2.5Fin':72, 'Gable':73,

                       'Hip':74, 'Gambrel':75, 'Mansard':76, 'Flat':77,'Shed':78, 'CompShg':79, 'WdShngl':80, 'Metal':81,

                       'WdShake':82, 'Membran':83,'Tar&Grv':84, 'Roll':85, 'ClyTile':86, 'VinylSd':87, 'MetalSd':88,

                       'Wd Sdng':89,'HdBoard':90,'WdShing':91, 'CemntBd':92, 'Plywood':93, 'AsbShng':94, 'Stucco':95,

                       'BrkComm':96, 'AsphShn':97,'ImStucc':98, 'CBlock':99, 'Wd Shng':100, 'CmentBd':101,'Brk Cmn':102,

                       'Other':103, 'PConc':104, 'BrkTil':105, 'Wood':106, 'Slab':107, 'GasA':108,'GasW':109, 'Grav':110,

                       'Wall':111, 'OthW':112, 'Floor':113, 'Y':114, 'N':115, 'Typ':116, 'Min1':117,'Maj1':118, 'Min2':119,

                       'Maj2':120,'P':121, 'WD':122, 'New':123, 'COD':124, 'ConLD':125,'ConLI':126, 'CWD':127, 'ConLw':128,

                       'Con':129, 'Oth':130, 'Normal':131, 'Abnorml':132,'Partial':133, 'AdjLand':134, 'Alloca':135, 'Family':136})
train.select_dtypes(include ='object').columns
# to check if there are any null values left in the dataset

b = train.isna().sum()

b = pd.DataFrame(b,columns=['value'])

b[b['value']!=0]
X = train.drop('SalePrice',axis=1)

Y = train['SalePrice']

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

reg = LinearRegression()

reg.fit(X,Y)

Y_pred = reg.predict(X)

print(r2_score(Y,Y_pred))

print(mean_squared_error(Y,Y_pred)**(1/2))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(X,Y)

Y_predRF = regressor.predict(X)

print(r2_score(Y,Y_predRF))

print(mean_squared_error(Y,Y_predRF)**(1/2))
train_norm = (train - train.mean()) / (train.max() - train.min())

X_norm = train_norm.drop('SalePrice',axis=1)

Y_norm = train['SalePrice']

reg.fit(X_norm,Y_norm)

Y_pred_norm = reg.predict(X_norm)

print(r2_score(Y_norm,Y_pred_norm))

print(mean_squared_error(Y_norm,Y_pred_norm)**(1/2))
regressor.fit(X_norm,Y_norm)

Y_pred_normRF = regressor.predict(X_norm)

print(r2_score(Y_norm,Y_pred_normRF))

print(mean_squared_error(Y_norm,Y_pred_normRF)**(1/2))
from sklearn.preprocessing import StandardScaler 

sc = StandardScaler() 

X_train_SC = sc.fit_transform(train) 
from sklearn.decomposition import PCA 

pca = PCA(n_components = 5) 

X_train_PCA = pca.fit_transform(X_train_SC)
regressor.fit(X_train_PCA,Y)

Y_pred_norm_PCARF = regressor.predict(X_train_PCA)

print(r2_score(Y,Y_pred_norm_PCARF))

print(mean_squared_error(Y,Y_pred_norm_PCARF)**(1/2))