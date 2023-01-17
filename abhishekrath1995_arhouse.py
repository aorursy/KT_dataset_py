import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as sns 

from sklearn.impute import KNNImputer

from sklearn.feature_selection import SelectFromModel

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split



from sklearn import metrics

from sklearn.preprocessing import scale 

from sklearn import model_selection

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error



path1 = "../input/house-prices-advanced-regression-techniques/train.csv"

train = pd.read_csv(path1)

path2 = "../input/house-prices-advanced-regression-techniques/test.csv"

test = pd.read_csv(path2)
test.info()
train_new = train.select_dtypes('object').fillna("Unknown")
df = pd.get_dummies(train_new, columns=["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig",

                                "LandSlope", "Neighborhood", "Condition1", "Condition2","BldgType", "HouseStyle", 

                                "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", 

                                "Neighborhood", "Condition1", "Condition2", "ExterCond", "Foundation" , "BsmtQual", 

                                "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", 

                                "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", 

                                "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", 

                                "MiscFeature", "SaleType", "SaleCondition"], 

                        prefix=["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig",

                                "LandSlope", "Neighborhood", "Condition1", "Condition2","BldgType", "HouseStyle", 

                                "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", 

                                "Neighborhood", "Condition1", "Condition2", "ExterCond", "Foundation" , "BsmtQual", 

                                "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", 

                                "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", 

                                "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", 

                                "MiscFeature", "SaleType", "SaleCondition"])
train1 = pd.concat([train_new, df], axis=1)
train1.head()
train1 = train1.drop(columns = ["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig",

                                "LandSlope", "Neighborhood", "Condition1", "Condition2","BldgType", "HouseStyle", 

                                "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", 

                                "Neighborhood", "Condition1", "Condition2", "ExterCond", "Foundation" , "BsmtQual", 

                                "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", 

                                "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", 

                                "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", 

                                "MiscFeature", "SaleType", "SaleCondition" ])
train1.head()
train1.info()
train2 = train.select_dtypes(exclude='object')

train2.head()
imputer = KNNImputer(n_neighbors=10)

train2_imputed = imputer.fit_transform(train2)
train2 = pd.DataFrame(train2_imputed, columns = train2.columns)
train2.head()
train_hr = pd.concat([train1, train2], axis=1, join='inner')
train_hr.head()
train_hr.isnull().sum()
train_hr.info()
sns.heatmap(train_hr.corr())
X = train_hr.drop(['SalePrice'],axis = 1)

y = train_hr['SalePrice']
from sklearn import decomposition, datasets

from sklearn.preprocessing import StandardScaler
# Create a scaler object

sc = StandardScaler()

# Fit the scaler to the features and transform

X_std = sc.fit_transform(X)
n_components = 30

pca = decomposition.PCA(n_components)
# Fit the PCA and transform the data

X_std_pca = pca.fit_transform(X_std)
X_std_pca
X_train, X_valid, y_train, y_valid = train_test_split(X_std_pca, y, test_size=0.33, random_state=123)


from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()

reg.fit(X_train,y_train)
#Predicting output

y_pred = reg.predict(X_valid)

y_pred
r2_score(y_valid, y_pred)

mean_squared_error(y_valid, y_pred)
test.head()
test_new = test.select_dtypes('object').fillna("Unknown")

df = pd.get_dummies(test_new, columns=["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig",

                                "LandSlope", "Neighborhood", "Condition1", "Condition2","BldgType", "HouseStyle", 

                                "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", 

                                "Neighborhood", "Condition1", "Condition2", "ExterCond", "Foundation" , "BsmtQual", 

                                "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", 

                                "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", 

                                "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", 

                                "MiscFeature", "SaleType", "SaleCondition"], 

                        prefix=["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig",

                                "LandSlope", "Neighborhood", "Condition1", "Condition2","BldgType", "HouseStyle", 

                                "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", 

                                "Neighborhood", "Condition1", "Condition2", "ExterCond", "Foundation" , "BsmtQual", 

                                "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", 

                                "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", 

                                "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", 

                                "MiscFeature", "SaleType", "SaleCondition"])

test1 = pd.concat([test_new, df], axis=1)

test1 = test1.drop(columns = ["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig",

                                "LandSlope", "Neighborhood", "Condition1", "Condition2","BldgType", "HouseStyle", 

                                "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", 

                                "Neighborhood", "Condition1", "Condition2", "ExterCond", "Foundation" , "BsmtQual", 

                                "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", 

                                "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", 

                                "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", 

                                "MiscFeature", "SaleType", "SaleCondition" ])
test2 = test.select_dtypes(exclude='object')

imputer = KNNImputer(n_neighbors=10)

test2_imputed = imputer.fit_transform(test2)

test2 = pd.DataFrame(test2_imputed, columns = test2.columns)

test_hr = pd.concat([test1, test2], axis=1, join='inner')
test_hr.head()
test_std = sc.fit_transform(test_hr.drop('Id', axis=1))

n_components = 30

pca = decomposition.PCA(n_components)

test_std_pca = pca.fit_transform(test_std)
# submit your predictions in csv format

Ids = test['Id']

predictions = reg.predict(test_std_pca)

output = pd.DataFrame({ 'Id' : Ids, 'SalePrice': predictions })

output.to_csv('submission.csv', index=False)