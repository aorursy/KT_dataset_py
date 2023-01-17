%pylab inline

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# reading the test files

data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")

from sklearn import pipeline, compose, impute, preprocessing



# Categorizing every feature and creating the pipelines

all_features = ['Id', 'MSSubClass', 'MSZoning', 

                'LotFrontage', 'LotArea', 'Street',

                'Alley', 'LotShape', 'LandContour', 

                'Utilities', 'LotConfig','LandSlope', 

                'Neighborhood', 'Condition1', 'Condition2', 

                'BldgType','HouseStyle', 'OverallQual', 

                'OverallCond', 'YearBuilt', 'YearRemodAdd',

                'RoofStyle', 'RoofMatl', 'Exterior1st', 

                'Exterior2nd', 'MasVnrType','MasVnrArea', 

                'ExterQual', 'ExterCond', 'Foundation', 

                'BsmtQual','BsmtCond', 'BsmtExposure', 

                'BsmtFinType1', 'BsmtFinSF1','BsmtFinType2',

                'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 

                'Heating','HeatingQC', 'CentralAir', 

                'Electrical', '1stFlrSF', '2ndFlrSF',

                'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',

                'BsmtHalfBath', 'FullBath','HalfBath', 

                'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

                'TotRmsAbvGrd', 'Functional', 'Fireplaces', 

                'FireplaceQu', 'GarageType','GarageYrBlt', 

                'GarageFinish', 'GarageCars', 'GarageArea', 

                'GarageQual','GarageCond', 'PavedDrive', 

                'WoodDeckSF', 'OpenPorchSF','EnclosedPorch',

                '3SsnPorch', 'ScreenPorch', 'PoolArea', 

                'PoolQC','Fence', 'MiscFeature', 'MiscVal', 

                'MoSold', 'YrSold', 'SaleType',

                'SaleCondition', 'SalePrice']



numeric_features = ["LotFrontage", "LotArea","OverallQual",

                    "OverallCond", "YearBuilt", "YearRemodAdd",

                    "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF",

                    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",

                    "LowQualFinSF", "GrLivArea", "BsmtFullBath",

                    "BsmtHalfBath", "FullBath", "HalfBath",

                    "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",

                    "Fireplaces", "GarageCars", "GarageArea",

                    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",

                    "3SsnPorch", "ScreenPorch", "PoolArea",

                    "MiscVal", "MoSold", "YrSold","GarageYrBlt",

                    "MasVnrArea"]



categorical_features = ["MSZoning", "Street", "Alley", "LotShape",

                       "LandContour", "Utilities", "LotConfig",

                       "LandSlope", "Neighborhood", "Condition1",

                       "Condition2", "BldgType", "HouseStyle",

                       "RoofStyle", "RoofMatl", "Exterior1st",

                       "Exterior2nd", "ExterQual",

                       "ExterCond", "Foundation", "BsmtQual",

                       "BsmtCond", "BsmtExposure","BsmtFinType1",

                       "BsmtFinType2", "Heating", "HeatingQC",

                       "CentralAir", "Electrical","KitchenQual",

                       "Functional", "FireplaceQu", "GarageType",

                       "GarageFinish", "GarageQual",

                       "GarageCond", "PavedDrive", "PoolQC",

                       "Fence", "MiscFeature", "SaleType",

                       "SaleCondition"]

                       

# cleaning the data and creating the transformers

numeric_cleanup = pipeline.make_pipeline(

    impute.SimpleImputer(strategy="median"),

 preprocessing.StandardScaler())



categorical_cleanup = pipeline.make_pipeline(

  impute.SimpleImputer(strategy="constant", fill_value="NA"),

  preprocessing.OneHotEncoder(handle_unknown="ignore"))



cleanup = compose.make_column_transformer(

  (numeric_cleanup, numeric_features),

  (categorical_cleanup, categorical_features))
from sklearn import model_selection

# cleanup data through pipeline use



cleanup.fit(data_train)



clean_train_X = cleanup.transform(data_train)

clean_test = cleanup.transform(data_test)

Y = data_train['SalePrice']



X_train, X_val, y_train, y_val = model_selection.train_test_split(clean_train_X, Y, test_size =0.25,random_state =7)
params = {

       'max_depth': [3, 4, 5, 6, 7],

        }
from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import TruncatedSVD

from sklearn import linear_model

from mlxtend.regressor import StackingCVRegressor



np.random.seed(786)



#Convert data to xgb form

data_dmatrix = xgb.DMatrix(data=clean_train_X,label= Y)



#Make Model

model1 = xgb.XGBRegressor(n_estimators = 5000, min_child_weight =1, subsample=.8, colsample_bytree=.6, gamma = .5, learning_rate=0.03, max_depth =3)

model2 = linear_model.LassoCV(cv = 3)

model3 = linear_model.RidgeCV(cv = 3)

model4 = linear_model.LinearRegression()

model5 = linear_model.SGDRegressor()

model6 = linear_model.LarsCV(cv = 3)



stack = StackingCVRegressor(regressors=(model1, model2, model3, model4, model5,model5,model6),

                            meta_regressor=model2)



stack.fit(clean_train_X.toarray(),Y)



#Test Validation Set

#y_pred = stack.predict(X_val)

#print("The RMSE is: {}".format(sqrt(mean_squared_log_error(y_val, y_pred ))))



#USED TO SUBMIT



y_pred = stack.predict(clean_test)



submission = pd.DataFrame({

    "Id": data_test["Id"],

   "SalePrice": y_pred

 })

submission.to_csv("submission.csv", index=False)
