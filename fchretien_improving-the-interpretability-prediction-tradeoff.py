import os

os.chdir("/kaggle/input/house-prices-advanced-regression-techniques/")  



import warnings

warnings.simplefilter(action='ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import uniform, randint, norm

import xgboost as xgb



from sklearn.preprocessing import OneHotEncoder, scale, StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.compose import make_column_transformer, TransformedTargetRegressor

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error



from math import sqrt      



# Installing Microsoft's package for "explanable boosting machine"

!pip install -U interpret



# Set random seed

np.random.seed(123)



# loading data

data = pd.read_csv("train.csv")
# droping (almost) perfectly correlated variables

data.drop(["GarageArea", "1stFlrSF", "GrLivArea"], axis=1)



# replacing intended NAs

NA_to_no = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", 

            "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", 

            "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]



for i in NA_to_no:

  data[i] = data[i].fillna("N")



# Droping the two features with many missing values

data = data.drop(["LotFrontage", "GarageYrBlt"], axis = 1)



#Dropping the outliers

data = data[data.GrLivArea<4000]



# Splitting the features from the target, and the train and test sets



X = data

X = X.drop("SalePrice", axis=1)

y = data.loc[:,"SalePrice"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)



# identifying the categorical and numeric variables



categorical = ["MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig","LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "MoSold", "SaleType", "SaleCondition", "BedroomAbvGr", "KitchenAbvGr"]



numeric = ["LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "2ndFlrSF", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "YrSold"]
# I use the log transformation for prediction



def log(x):

    return np.log(x)



def exp(x):

    return np.exp(x)



# Setting up the preprocessor.

preprocessor = make_column_transformer((make_pipeline(SimpleImputer(strategy="most_frequent"), 

                                                      OneHotEncoder(handle_unknown="ignore")), categorical), 

                                       (make_pipeline(SimpleImputer(strategy="median"),

                                                      StandardScaler()), numeric))



# Instantiating the model

pipeline_linear = make_pipeline(preprocessor,

                               TransformedTargetRegressor(LinearRegression(),

                               func=log, inverse_func =exp))



#Fitting the model and retrieving the prediction

pipeline_linear.fit(X_train, y_train)

line_pred = pipeline_linear.predict(X_test)
pipeline_xgb = make_pipeline(preprocessor,

                    TransformedTargetRegressor(xgb.XGBRegressor(objective ='reg:squarederror', nthread=-1), 

                                               func=log, inverse_func=exp))

# Hyperparameters distributions

params = {

    "transformedtargetregressor__regressor__colsample_bytree": uniform(0.7, 0.3),

    "transformedtargetregressor__regressor__gamma": uniform(0, 0.5),

    "transformedtargetregressor__regressor__learning_rate": uniform(0.03, 0.3),

    "transformedtargetregressor__regressor__max_depth": randint(2, 6),

    "transformedtargetregressor__regressor__n_estimators": randint(500, 1000),

    "transformedtargetregressor__regressor__subsample": uniform(0.6, 0.4)

}



# Instantiating the xgboost model, with random-hyperparameter tuning

xgb_model = RandomizedSearchCV(pipeline_xgb, param_distributions=params, random_state=123, 

                               n_iter=50, cv=5, n_jobs=-1)



#Fitting the model and retrieving the predictions

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
from interpret.glassbox import ExplainableBoostingRegressor

from interpret import show

from interpret.data import Marginal



# Definition of the EBM preprocessor; I do not one hot encode, since EBM deals with categoricals



preprocessor_ebm = make_column_transformer(

    (SimpleImputer(strategy="most_frequent"), categorical),

    (SimpleImputer(strategy="median"), numeric)

    )



# Instantiating the model

ebm = make_pipeline(preprocessor_ebm, 

                    TransformedTargetRegressor(ExplainableBoostingRegressor(random_state=123),

                    func=log, inverse_func=exp))



#Fitting the model and retrieving the predictions

ebm.fit(X_train, y_train)

ebm_pred = ebm.predict(X_test)
params = {

    "xgbregressor__colsample_bytree": uniform(0.7, 0.3),

    "xgbregressor__gamma": uniform(0, 0.5),

    "xgbregressor__learning_rate": uniform(0.03, 0.3),

    "xgbregressor__max_depth": randint(2, 6),

    "xgbregressor__n_estimators": randint(500, 1000),

    "xgbregressor__subsample": uniform(0.6, 0.4)

}



pipeline_xgb2 = make_pipeline(preprocessor,

                              xgb.XGBRegressor(objective ='reg:squarederror', nthread=-1))



xgb_model_2 = RandomizedSearchCV(pipeline_xgb2, param_distributions=params, random_state=123,

                                 n_iter=50, cv=5)



# getting residual predictions from the train data

ebm_pred_train = ebm.predict(X_train)

ebm_residual_train = y_train - ebm_pred_train



# training the xgb from the train data residual

xgb_model_2.fit(X_train, ebm_residual_train)

residual_predicted = xgb_model_2.predict(X_test)



# then we get our boosted ebm prediction

ebm_xgb_pred = ebm_pred + residual_predicted
# Getting performance 



predict = [line_pred, xgb_pred, ebm_pred, ebm_xgb_pred]



mae = []

mse = []

rmse = []



for i in predict:

    mae.append(mean_absolute_error(y_test, i))

    mse.append(mean_squared_error(y_test, i))

    rmse.append(sqrt(mean_squared_error(y_test, i)))



scores = pd.DataFrame([mae, mse, rmse], 

                      columns=["line", "xgb", "ebm", "ebm + xgb"],

                      index = ["mae", "mse", "rmse"])



scores["ebm + xgb over ebm"] = (round((scores["ebm"]/scores["ebm + xgb"] -1)*100, 2)\

                                .astype(str) +" %")

scores["xgb over ebm + xgb"] = (round((1- scores["xgb"]/scores["ebm + xgb"])*100, 2)\

                                .astype(str) +" %")



scores