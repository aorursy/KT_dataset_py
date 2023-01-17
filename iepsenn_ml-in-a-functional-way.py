import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/train.csv")

df.sample(5)
df.describe()
df.columns
!pip install fklearn
columns_to_train = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea", 

 "TotalBsmtSF", "1stFlrSF", "FullBath", "TotRmsAbvGrd", 

 "YearBuilt", "Neighborhood", 'KitchenQual', 'HouseStyle', 'LotArea', 'SalePrice']
train = df[columns_to_train]

train.dtypes
from fklearn.tuning.model_agnostic_fc import correlation_feature_selection



correlation_feature_selection(train,features=columns_to_train)['feature_corr']['SalePrice'] #threshold)
train.isnull().sum()
import matplotlib.pyplot as plt 

columns_to_explore = ["GrLivArea", "GarageArea", 

 "TotalBsmtSF", "1stFlrSF", "TotRmsAbvGrd", 

 "YearBuilt", 'LotArea', 'SalePrice']



#eplore 3 by 3

train.boxplot(column=columns_to_explore[:3])
train.boxplot(column=columns_to_explore[3:6])
train.boxplot(column=columns_to_explore[6:])
train[train.GrLivArea < 2650].boxplot(column="GrLivArea")

train = train.drop(train[train.GrLivArea > 2650].index).copy()
train[train.GarageArea < 950].boxplot(column="GarageArea")

train = train.drop(train[train.GarageArea > 950].index).copy()
train[(train.TotalBsmtSF < 1800) & (train.TotalBsmtSF > 200)].boxplot(column="TotalBsmtSF")

train = train.drop(train[(train.TotalBsmtSF > 1800) | (train.TotalBsmtSF < 200)].index).copy()
train[train['1stFlrSF'] < 1850].boxplot(column="1stFlrSF")

train = train.drop(train[train['1stFlrSF'] > 1850].index).copy()
train[train.TotRmsAbvGrd < 10].boxplot(column="TotRmsAbvGrd")

train = train.drop(train[train.TotRmsAbvGrd > 10].index).copy()
train[train.YearBuilt > 1885].boxplot(column="YearBuilt")

train = train.drop(train[train.YearBuilt < 1885].index).copy()
train[(train.SalePrice < 280000) & (train.SalePrice > 45000)].boxplot(column='SalePrice')

train = train.drop(train[(train.SalePrice > 280000) | (train.SalePrice < 45000)].index).copy()
train[(train.LotArea < 15800) & (train.LotArea > 2100)].boxplot(column="LotArea")

train = train.drop(train[(train.LotArea > 15800) | (train.LotArea < 2100)].index).copy()
from fklearn.training.pipeline import build_pipeline

from fklearn.training.regression import linear_regression_learner

from fklearn.training.transformation import capper, floorer, prediction_ranger, label_categorizer

from fklearn.training.imputation import imputer



def fit(train_data):

    labelcateg_fn = label_categorizer(columns_to_categorize=['Neighborhood', 'KitchenQual', 'HouseStyle'])

    imputer_fn = imputer(columns_to_impute=columns_to_train[:-1])

    capper_fn = capper(columns_to_cap=columns_to_train[:-1])

    regression_fn = linear_regression_learner(features=columns_to_train[:-1], 

                                              target=columns_to_train[-1], 

                                              prediction_column="Predicted_Value")

    

    learner = build_pipeline(labelcateg_fn, imputer_fn, capper_fn, regression_fn)

    predict_fn, training_predictions, logs = learner(train_data)

    

    return predict_fn, logs
from math import ceil



n_train = ceil(0.75*train.shape[0])



train_df = train.iloc[:n_train,:]

test_df = train.iloc[n_train:,:]



model, log = fit(train_df)
test_df.describe()
preds = model(test_df)
preds
from fklearn.validation.evaluators import mse_evaluator, mean_prediction_evaluator



mse_evaluator(preds, target_column="SalePrice", prediction_column="Predicted_Value")
mean_prediction_evaluator(preds, prediction_column="Predicted_Value")
to_predict = pd.read_csv("../input/test.csv")

to_predict.sample(5)
index = to_predict.Id

print(train.columns)

to_predict = to_predict[train.columns[:-1]]

to_predict.isnull().sum() 
to_predict.describe()
to_predict.GarageCars.fillna(to_predict.GarageCars.mean(), inplace=True)

to_predict.GarageArea.fillna(to_predict.GarageArea.mean(), inplace=True)

to_predict.TotalBsmtSF.fillna(to_predict.TotalBsmtSF.mean(), inplace=True)

to_predict.KitchenQual.fillna("TA", inplace=True)

to_predict.isnull().sum() 
preds_ = model(to_predict)

predictions_ = preds_['Predicted_Value'].values

predictions_
d = {"Id" : index, "SalePrice" : predictions_}

house_prices = pd.DataFrame(data=d)

house_prices.to_csv("house_prices.csv", index=False)