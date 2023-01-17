import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.info()
def clean_data_by_zero(dataset, columns):

    for column in columns:

        dataset[column] = dataset[column].fillna(0)

    return dataset

    
train = clean_data_by_zero(train, ['BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

                          'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',

                          'PoolQC','Fence','MiscFeature'])
train['LotFrontage'].fillna((train['LotFrontage'].median()),inplace=True)

train['MasVnrArea'].fillna((train['LotFrontage'].median()),inplace=True)
train = train.drop(['LotShape','LandContour','LotConfig',

                    'Neighborhood', 'Condition1', 'GarageYrBlt',

                    'HouseStyle','MSZoning', 'SaleCondition'], axis=1)

train = train.drop(['RoofStyle','Alley','Exterior1st',

                    'Exterior2nd','MasVnrType','Heating',

                    'Electrical','Functional', 'SaleType'], axis=1)
street_df = pd.get_dummies(train['Street'],prefix='Street')

train = train.join(street_df)

train = train.drop(['Street'],axis=1)
train['Utilities'] = train['Utilities'].apply({'AllPub':1,'NoSeWa':0}.get)

train['LandSlope'] = train['LandSlope'].apply({'Gtl':3,'Mod':2,'Sev':1}.get)

train['Condition2'] = train['Condition2'].apply(lambda x: 1 if x == 'Norm' else 0)

train['BldgType'] = train['BldgType'].apply({'1Fam':3,'2fmCon':2,'Duplex':2, 'TwnhsE':1, 'Twnhs':1}.get)

train['RoofMatl'] = train['RoofMatl'].apply({'CompShg':4, 'WdShngl':3, 'Metal':3, 'WdShake':3,'Membran':2, 'Tar&Grv':2,'Roll':1, 'ClyTile':1}.get)

train['ExterQual'] = train['ExterQual'].apply({'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po': 1}.get)

train['ExterCond'] = train['ExterCond'].apply({'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po':1}.get)

train['Foundation'] = train['Foundation'].apply({'BrkTil':5,'CBlock':5, 'PConc':4, 'Slab':3, 'Stone':2, 'Wood':1}.get)

train['BsmtQual'] = train['BsmtQual'].apply({'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 0:0}.get)

train['BsmtCond'] = train['BsmtCond'].apply({'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 0:0}.get)

train['BsmtExposure'] = train['BsmtExposure'].apply({'Gd':4,'Av':3, 'Mn':2, 'No':1, 0:0}.get)

train['BsmtFinType1'] = train['BsmtFinType1'].apply({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1, 0:0}.get)

train['BsmtFinType2'] = train['BsmtFinType2'].apply({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1, 0:0}.get)

train['HeatingQC'] = train['HeatingQC'].apply({'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po':1}.get)

train['CentralAir'] = train['CentralAir'].apply({'Y':1,'N':0}.get)

train['KitchenQual'] = train['KitchenQual'].apply({'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po':1}.get)

train['FireplaceQu'] =  train['FireplaceQu'].apply({'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 0:0}.get)

train['GarageType'] = train['GarageType'].apply({'2Types':6,

                                                 'Attchd':5, 

                                                 'Basment':4,

                                                 'BuiltIn':3, 

                                                 'CarPort':2,

                                                 'Detchd':1,

                                                 0:0}.get)

train['GarageFinish'] = train['GarageFinish'].apply({'Fin':3,

                                                 'RFn':2, 

                                                 'Unf':1,

                                                 0:0}.get)

train['GarageCond'] = train['GarageCond'].apply({'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 0:0}.get)

train['GarageQual'] = train['GarageQual'].apply({'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 0:0}.get)

train['PavedDrive'] = train['PavedDrive'].apply({'Y':2,'P':1,'N':0}.get)

train['PoolQC']= train['PoolQC'].apply({'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 0:0}.get)

train['Fence'] = train['Fence'].apply({'GdPrv':4, 'MnPrv': 3, 'GdWo': 2, 'MnWw':1, 0:0 }.get)

train['MiscFeature'] = train['MiscFeature'].apply(lambda x: 1 if x != 0 else 0)
train.head()
y = train[['SalePrice']]

X = train.loc[:, train.columns != 'SalePrice']
from sklearn.model_selection import train_test_split
# Create the training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)
import xgboost as xgb



# Instantiate the XGBRegressor as xg_reg

xg_reg = xgb.XGBRegressor(seed=123, objective="reg:linear", n_estimators=10)
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
# import the mean_squared_error from sklearn library

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, preds))
rmse = np.sqrt(mean_squared_error(y_test,preds))

print("RMSE: %f" % (rmse))