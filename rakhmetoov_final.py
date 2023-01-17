import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df_train.sample(5)
df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
df_test.sample(5)
df_train.shape

df_train.info()
y_train = df_train['SalePrice'].values
y_train
df_train.hist(column='SalePrice', bins=20)
df_train2 = df_train.copy()

df_train2['MSZoning'] =df_train2['MSZoning'].astype('category').cat.codes
df_train2['Street'] =df_train2['Street'].astype('category').cat.codes
df_train2['Alley'] =df_train2['Alley'].astype('category').cat.codes
df_train2['LotShape'] =df_train2['LotShape'].astype('category').cat.codes
df_train2['LandContour'] =df_train2['LandContour'].astype('category').cat.codes
df_train2['Utilities'] =df_train2['Utilities'].astype('category').cat.codes
df_train2['LotConfig'] =df_train2['LotConfig'].astype('category').cat.codes
df_train2['LandSlope'] =df_train2['LandSlope'].astype('category').cat.codes
df_train2['Neighborhood'] =df_train2['Neighborhood'].astype('category').cat.codes
df_train2['Condition1'] =df_train2['Condition1'].astype('category').cat.codes
df_train2['Condition2'] =df_train2['Condition2'].astype('category').cat.codes
df_train2['BldgType'] =df_train2['BldgType'].astype('category').cat.codes
df_train2['HouseStyle'] =df_train2['HouseStyle'].astype('category').cat.codes
df_train2['RoofStyle'] =df_train2['RoofStyle'].astype('category').cat.codes
df_train2['RoofMatl'] =df_train2['RoofMatl'].astype('category').cat.codes
df_train2['Exterior1st'] =df_train2['Exterior1st'].astype('category').cat.codes
df_train2['Exterior2nd'] =df_train2['Exterior2nd'].astype('category').cat.codes
df_train2['MasVnrType'] =df_train2['MasVnrType'].astype('category').cat.codes
df_train2['ExterQual'] =df_train2['ExterQual'].astype('category').cat.codes
df_train2['MasVnrType'] =df_train2['MasVnrType'].astype('category').cat.codes
df_train2['ExterCond'] =df_train2['ExterCond'].astype('category').cat.codes
df_train2['Foundation'] =df_train2['Foundation'].astype('category').cat.codes
df_train2['BsmtQual'] =df_train2['BsmtQual'].astype('category').cat.codes
df_train2['BsmtCond'] =df_train2['BsmtCond'].astype('category').cat.codes
df_train2['BsmtExposure'] =df_train2['BsmtExposure'].astype('category').cat.codes
df_train2['BsmtFinType1'] =df_train2['BsmtFinType1'].astype('category').cat.codes
df_train2['BsmtFinType2'] =df_train2['BsmtFinType2'].astype('category').cat.codes
df_train2['Heating'] =df_train2['Heating'].astype('category').cat.codes
df_train2['HeatingQC'] =df_train2['HeatingQC'].astype('category').cat.codes
df_train2['CentralAir'] =df_train2['CentralAir'].astype('category').cat.codes
df_train2['Electrical'] =df_train2['Electrical'].astype('category').cat.codes
df_train2['KitchenQual'] =df_train2['KitchenQual'].astype('category').cat.codes
df_train2['Functional'] =df_train2['Functional'].astype('category').cat.codes
df_train2['FireplaceQu'] =df_train2['FireplaceQu'].astype('category').cat.codes
df_train2['GarageType'] =df_train2['GarageType'].astype('category').cat.codes
df_train2['GarageFinish'] =df_train2['GarageFinish'].astype('category').cat.codes
df_train2['GarageQual'] =df_train2['GarageQual'].astype('category').cat.codes
df_train2['GarageCond'] =df_train2['GarageCond'].astype('category').cat.codes
df_train2['PavedDrive'] =df_train2['PavedDrive'].astype('category').cat.codes
df_train2['PoolQC'] =df_train2['PoolQC'].astype('category').cat.codes
df_train2['Fence'] =df_train2['Fence'].astype('category').cat.codes
df_train2['MiscFeature'] =df_train2['MiscFeature'].astype('category').cat.codes
df_train2['SaleType'] =df_train2['SaleType'].astype('category').cat.codes
df_train2['SaleCondition'] =df_train2['SaleCondition'].astype('category').cat.codes

df_train2.corr()['SalePrice'].sort_values(ascending=False)[30:70]
df_train = df_train.drop(columns = ['BsmtUnfSF', 'SaleCondition', 'Neighborhood', 'HouseStyle', 'BedroomAbvGr', 'BsmtCond', 'RoofMatl', 'BsmtFinType2', 'ExterCond', 
                                    'Functional', 'ScreenPorch', 'Exterior2nd', 'Exterior1st', 'PoolArea', 'Condition1', 'PoolQC', 'LandSlope', 'MoSold', '3SsnPorch',
                                   'Street', 'LandContour', 'Condition2', 'MasVnrType', 'BsmtFinSF2', 'BsmtFinType1', 'Utilities', 'BsmtHalfBath', 'MiscVal', 'Id',
                                   'LowQualFinSF', 'YrSold', 'SaleType', 'LotConfig', 'MiscFeature', 'OverallCond', 'MSSubClass', 'BldgType', 'Alley', 'Heating', 'EnclosedPorch'])
df_train.head()
df_train = df_train.iloc[:,:40]
df_train.head()
id_column = df_test['Id']
df_test = df_test.drop(columns = ['BsmtUnfSF', 'SaleCondition', 'Neighborhood', 'HouseStyle', 'BedroomAbvGr', 'BsmtCond', 'RoofMatl', 'BsmtFinType2', 'ExterCond', 
                                    'Functional', 'ScreenPorch', 'Exterior2nd', 'Exterior1st', 'PoolArea', 'Condition1', 'PoolQC', 'LandSlope', 'MoSold', '3SsnPorch',
                                   'Street', 'LandContour', 'Condition2', 'MasVnrType', 'BsmtFinSF2', 'BsmtFinType1', 'Utilities', 'BsmtHalfBath', 'MiscVal', 'Id',
                                   'LowQualFinSF', 'YrSold', 'SaleType', 'LotConfig', 'MiscFeature', 'OverallCond', 'MSSubClass', 'BldgType', 'Alley', 'Heating', 'EnclosedPorch'])
df_test.head()
df_train.dtypes
df_train_encoded = pd.get_dummies(df_train,columns=['MSZoning', 'LotShape', 'RoofStyle', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir', 
                                                    'Electrical', 'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'Fence'
                                                   ])
df_train_encoded.head()
df_test_encoded = pd.get_dummies(df_test,columns=['MSZoning', 'LotShape', 'RoofStyle', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir', 
                                                    'Electrical', 'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'Fence'
                                                   ])
df_test_encoded.head()
for col1 in df_train_encoded.columns: 
    ok = False
    for col2 in df_test_encoded.columns: 
        if col1 == col2:
            ok = True
            
    if ok == False:
        df_test_encoded[col1] = 0

df_test_encoded.head()
df_train_encoded = df_train_encoded.fillna(df_train_encoded.mean())
df_train_encoded.head()
df_test_encoded = df_test_encoded.fillna(df_test_encoded.mean())
df_test_encoded.head()
X_train = df_train_encoded.values
X_test = df_test_encoded.values
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred
from sklearn.linear_model import LogisticRegression

log_reg_cls = LogisticRegression()
log_reg_cls.fit(X_train, y_train)
y_pred = log_reg_cls.predict(X_test)
y_pred
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)
y_pred = clf.fit(X_train, y_train).predict(X_test)
import xgboost as xgb

xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed=123 )
xg_cl.fit(X_train, y_train)
y_pred = xg_cl.predict(X_test)
result = pd.concat([id_column, pd.Series(y_pred)], axis = 1)
result.to_csv('result9.csv', index=False)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

knn = KNeighborsClassifier(11)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)