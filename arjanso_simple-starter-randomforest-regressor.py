import pandas as pd

import numpy as np

import os

import re

import random

import datetime

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
random.seed(3)  #To make the randomization reproducible

pd.options.mode.chained_assignment = None  #To turn off specific warnings
train = pd.read_csv(r"../input/train_2016_v2.csv")   #The parcelid's with their outcomes

props = pd.read_csv(r"../input/properties_2016.csv")  #The properties dataset

samp = pd.read_csv(r"../input/sample_submission.csv")  #The parcelid's for the testset
props = props.select_dtypes(exclude=[object])  #For this example, we take only numerical data, since strings require more processing

props.fillna(-1,inplace=True)  #Fill missing data so we can run the model

train = train.loc[:,['parcelid','logerror']].merge(props,how='left',left_on='parcelid',right_on='parcelid')

train_x = train.drop(['parcelid','logerror'],axis=1,inplace=False)

train_y = train['logerror']



test = samp.loc[:,['ParcelId']].merge(props,how='left',left_on='ParcelId',right_on='parcelid')

test_x = test.drop(['ParcelId','parcelid'],axis=1,inplace=False)
parameters = {'n_estimators':[5,10,15],'n_jobs':[-1],'oob_score':[False]}  # this can be extended

model = RandomForestRegressor()

grid = GridSearchCV(model,param_grid=parameters,scoring='neg_mean_absolute_error',cv=3)  

grid.fit(train_x,train_y)
cv_results = pd.DataFrame(grid.cv_results_)

print(cv_results[["param_n_estimators","mean_test_score","std_test_score"]])



feat_imps = grid.best_estimator_.feature_importances_

fi = pd.DataFrame.from_dict({'feat':train_x.columns,'imp':feat_imps})

fi.set_index('feat',inplace=True,drop=True)

fi = fi.sort_values('imp',ascending=False)

fi.head(20).plot.bar()
test_y = grid.predict(test_x)

test_y = pd.DataFrame(test_y)

test_y[1] = test_y[0]

test_y[2] = test_y[0]

test_y[3] = test_y[0]

test_y[4] = test_y[0]

test_y[5] = test_y[0]  #For simplicity make identical predictions for all months

test_y.columns = ["201610","201611","201612","201710","201711","201712"]

submission = test_y.copy()

submission["parcelid"] = samp["ParcelId"].copy()

cols = ["parcelid","201610","201611","201612","201710","201711","201712"]

submission = submission[cols]

filename = "Prediction_" + str(submission.columns[0]) + re.sub("[^0-9]", "",str(datetime.datetime.now())) + '.csv'

print(filename)

submission.to_csv(filename,index=False)
