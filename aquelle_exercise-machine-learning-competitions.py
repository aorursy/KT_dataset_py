# Code you have previously used to load data
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
#utility functions
def outliers(df):
        return pd.DataFrame(
            [
                df.LotFrontage > 250,
                df.BsmtFinSF1 > 4000,
                df.TotalBsmtSF > 6000,
                df['1stFlrSF'] > 4000,
                pd.DataFrame([
                    df.GrLivArea > 4000, 
                    np.log(df.SalePrice) < 13]).all(),
                pd.DataFrame([
                    df.GarageArea > 1200, 
                    np.log(df.SalePrice) < 12.5]).all(),
                pd.DataFrame([
                    df.OpenPorchSF > 500, 
                    np.log(df.SalePrice) < 11]).all(),
            ]
        ).any()
        
def removeOutliers(df):
    return df[~outliers(df)]

# Read training data
iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)
# Read test data
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)
#drop columns
home_data = home_data.drop('MiscFeature', axis = 1)
test_data = test_data.drop('MiscFeature', axis = 1)
#imputation
home_data["PoolQC"] = home_data["PoolQC"].fillna("None")
home_data["Alley"] = home_data["Alley"].fillna("None")
home_data["Fence"] = home_data["Fence"].fillna("None")
home_data["FireplaceQu"] = home_data["FireplaceQu"].fillna("None")
home_data["LotFrontage"] = home_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    home_data[col] = home_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    home_data[col] = home_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    home_data[col] = home_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    home_data[col] = home_data[col].fillna('None')
home_data["MasVnrType"] = home_data["MasVnrType"].fillna("None")
home_data["MasVnrArea"] = home_data["MasVnrArea"].fillna(0)
home_data['MSZoning'] = home_data['MSZoning'].fillna(home_data['MSZoning'].mode()[0])
home_data["Functional"] = home_data["Functional"].fillna("Typ")
home_data['Electrical'] = home_data['Electrical'].fillna(home_data['Electrical'].mode()[0])
home_data['KitchenQual'] = home_data['KitchenQual'].fillna(home_data['KitchenQual'].mode()[0])
home_data['Exterior1st'] = home_data['Exterior1st'].fillna(home_data['Exterior1st'].mode()[0])
home_data['Exterior2nd'] = home_data['Exterior2nd'].fillna(home_data['Exterior2nd'].mode()[0])
home_data['SaleType'] = home_data['SaleType'].fillna(home_data['SaleType'].mode()[0])
home_data['MSSubClass'] = home_data['MSSubClass'].fillna("None")
home_data['TotalSF'] = home_data['TotalBsmtSF'] + home_data['1stFlrSF'] + home_data['2ndFlrSF']

test_data["PoolQC"] = test_data["PoolQC"].fillna("None")
test_data["Alley"] = test_data["Alley"].fillna("None")
test_data["Fence"] = test_data["Fence"].fillna("None")
test_data["FireplaceQu"] = test_data["FireplaceQu"].fillna("None")
test_data["LotFrontage"] = test_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    test_data[col] = test_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    test_data[col] = test_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    test_data[col] = test_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    test_data[col] = test_data[col].fillna('None')
test_data["MasVnrType"] = test_data["MasVnrType"].fillna("None")
test_data["MasVnrArea"] = test_data["MasVnrArea"].fillna(0)
test_data['MSZoning'] = test_data['MSZoning'].fillna(test_data['MSZoning'].mode()[0])
test_data["Functional"] = test_data["Functional"].fillna("Typ")
test_data['Electrical'] = test_data['Electrical'].fillna(test_data['Electrical'].mode()[0])
test_data['KitchenQual'] = test_data['KitchenQual'].fillna(test_data['KitchenQual'].mode()[0])
test_data['Exterior1st'] = test_data['Exterior1st'].fillna(test_data['Exterior1st'].mode()[0])
test_data['Exterior2nd'] = test_data['Exterior2nd'].fillna(test_data['Exterior2nd'].mode()[0])
test_data['SaleType'] = test_data['SaleType'].fillna(test_data['SaleType'].mode()[0])
test_data['MSSubClass'] = test_data['MSSubClass'].fillna("None")
test_data['TotalSF'] = test_data['TotalBsmtSF'] + test_data['1stFlrSF'] + test_data['2ndFlrSF']
#Transform date data
home_data['YearBuilt'] = home_data['YrSold']-home_data['YearBuilt']
home_data['YearRemodAdd'] = home_data['YrSold']-home_data['YearRemodAdd']
test_data['YearBuilt'] = test_data['YrSold']-test_data['YearBuilt']
test_data['YearRemodAdd'] = test_data['YrSold']-test_data['YearRemodAdd']
home_data['GarageYrBlt'] = home_data['GarageYrBlt'] - home_data['YearBuilt']
test_data['GarageYrBlt'] = test_data['GarageYrBlt'] - test_data['YearBuilt']
#transform some categorical data
home_data['Street'] = home_data['Street'].map({'Pave': 0, 'Grvl': 1})
home_data['Utilities'] = home_data['Utilities'].map({'AllPub': 0, 'NoSeWa': 1})
home_data['CentralAir'] = home_data['CentralAir'].map({'N': 0, 'Y': 1})
home_data['LandSlope'] = home_data['LandSlope'].map({'Sev': 0, 'Gtl': 2, 'Mod': 1})
home_data['Alley'] = home_data['Alley'].map({'None': 0, 'Grvl': 1, 'Pave': 2})
test_data['Street'] = test_data['Street'].map({'Pave': 0, 'Grvl': 1})
test_data['Utilities'] = test_data['Utilities'].map({'AllPub': 0, 'NoSeWa': 1})
test_data['CentralAir'] = test_data['CentralAir'].map({'N': 0, 'Y': 1})
test_data['LandSlope'] = test_data['LandSlope'].map({'Sev': 0, 'Gtl': 2, 'Mod': 1})
test_data['Alley'] = test_data['Alley'].map({'None': 0, 'Grvl': 1, 'Pave': 2})
#Handle rest automatically
catCols = home_data.select_dtypes(include = 'object').columns.values
numCols = home_data.select_dtypes(include = [np.number]).columns.values
for col in catCols: 
    home_data[col] = home_data[col].astype('category').cat.as_ordered()
    test_data[col] = test_data[col].astype('category').cat.as_ordered()

testX = pd.get_dummies(test_data)
trainX = pd.get_dummies(home_data)
#remove outliers
noOutlierData = removeOutliers(trainX)
noOutlierY = noOutlierData.SalePrice
noOutlierX = pd.get_dummies(noOutlierData.drop('SalePrice', axis = 1))
# Create target object
#trainX = trainX.drop('SalePrice', axis = 1)
trainY = home_data.SalePrice
trainX, testX = trainX.align(testX, join = 'left', axis = 1)
trainX, noOutlierX = trainX.align(noOutlierX, join = 'left', axis = 1)

colsWithMissing = [col for col in trainX.columns
                      if trainX[col].isnull().any() or testX[col].isnull().any()]
#for col in colsWithMissing:
#    trainX[col+'_wasMissing'] = trainX[col].isnull()
#    testX[col+'_wasMissing'] = testX[col].isnull()
print(trainX.shape)
print(noOutlierX.shape)
#define imputer and impute missing values
paramGrid = {'nthread':[4],
#                       'objective':['reg:linear'],
                      'learning_rate': [.05], #so called `eta` value
                      'max_depth': [3],
                      'min_child_weight': [3],
                      'subsample': [.8],
                      'colsample_bytree': [.7],
                      'n_estimators': [980],
                      'reg_lambda' : [0.5,1,1.5,2,2.5,3]
                   }
gs_estimator = GridSearchCV(XGBRegressor(),
                            paramGrid,
                            cv = 5, scoring='neg_mean_absolute_error',
                            refit=True,
                            verbose=1)
def getBestTree():
    pipeline = make_pipeline(gs_estimator)
    pipeline.fit(noOutlierX, noOutlierY)
    return gs_estimator.best_estimator_

model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.7, gamma=0, learning_rate=0.05, max_delta_step=0,
       max_depth=3, min_child_weight=3, missing=None, n_estimators=980,
       n_jobs=1, nthread=1, objective='reg:linear', random_state=14,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=0.8)
pipeline = make_pipeline(model)
scores = cross_val_score(pipeline, noOutlierX, noOutlierY, cv = 5, scoring = 'neg_mean_absolute_error')

print(np.mean(scores))
model.fit(noOutlierX, noOutlierY)
test_preds = pipeline.predict(testX)
#The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)