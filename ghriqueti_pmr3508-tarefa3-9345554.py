from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
trainDS = pd.read_csv("../input/train.csv", names=["Id","latitude","longitude","median_age","total_rooms","total_bedrooms","population","households","median_income","median_house_value"], sep=r'\s*,\s*',engine='python',na_values='?', skiprows=1)
trainDS.shape
trainDS.head()
plt.scatter(trainDS["latitude"],trainDS["median_house_value"])
regr = linear_model.LinearRegression()
trainDS.isnull().values.sum()
trainDS_X = trainDS[["Id","latitude","longitude","median_age","total_rooms","total_bedrooms","population","households","median_income"]]
trainDS_Y = trainDS[["median_house_value"]]
regr.fit(trainDS_X,trainDS_Y)
testDS_X = pd.read_csv("../input/test.csv", names=["Id","latitude","longitude","median_age","total_rooms","total_bedrooms","population","households","median_income"], sep=r'\s*,\s*',engine='python',na_values='?', skiprows=1)
testDS_X.head()
testDS_X.isnull().values.sum()
testDS_Y = regr.predict(testDS_X)
print('Coefficients: \n', regr.coef_)
print('Interception: \n', regr.intercept_)
scores = cross_val_score(regr, trainDS_X, trainDS_Y, cv=10)
scores
regRCV = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100], cv=5)
regRCV.fit(trainDS_X,trainDS_Y)
regRCV.alpha_
regRCV = linear_model.RidgeCV(alphas=[6, 8, 10, 12, 14], cv=5)
regRCV.fit(trainDS_X,trainDS_Y)
regRCV.alpha_
regR = linear_model.Ridge(alpha = 10)
regR.fit(trainDS_X,trainDS_Y)
testDS_Y = regR.predict(testDS_X)
print('Coefficients: \n', regR.coef_)
print('Interception: \n', regR.intercept_)
scoresRidge = cross_val_score(regR, trainDS_X, trainDS_Y, cv=10)
scoresRidge
regLCV = linear_model.LassoCV(alphas=[0.1, 1, 10, 20, 100], cv=5)
regLCV.fit(trainDS_X,trainDS_Y["median_house_value"])
regLCV.alpha_
regLCV = linear_model.LassoCV(alphas=[15, 18, 20, 22, 24], cv=5)
regLCV.fit(trainDS_X,trainDS_Y["median_house_value"])
regLCV.alpha_
regL = linear_model.Lasso(alpha=22)
regL.fit(trainDS_X,trainDS_Y)
testDS_Y = regL.predict(testDS_X)
print('Coefficients: \n', regL.coef_)
print('Interception: \n', regL.intercept_)
scoresLasso = cross_val_score(regL, trainDS_X, trainDS_Y, cv=10)
scoresLasso
from numpy import linalg as LA
norm = LA.norm(scores)
normR = LA.norm(scoresRidge)
normL = LA.norm(scoresLasso)
norm
normR
normL
testDS_Predict = pd.DataFrame()
testDS_Predict["Id"] = testDS_X["Id"]
testDS_Predict["median_house_value"] = testDS_Y
testDS_Predict.head()
num = testDS_Predict._get_numeric_data()
num[num<0]=0
testDS_Predict.to_csv("testDS_3.csv", index=False)