import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sb



#function definitions

def EucDist(x1, y1, x2, y2):

    return np.sqrt( np.power((x1-x2),2) + np.power((y1-y2), 2) )



def DoGridSearchTuning(X, y, numFolds, params, model):

    gsRes = GridSearchCV(model, param_grid=params, cv=numFolds, n_jobs=-1, scoring='neg_mean_absolute_error')

    gsRes.fit(X,y)

    return gsRes



def PlotResultsOfGridSearch(gsResult):

    meanScores = np.absolute(gsResult.cv_results_['mean_test_score'])

    #print(meanScores)

    T = gsResult.cv_results_['params']

    axes = T[0].keys()

    cnt = 0

    var = []

    for ax in axes:

        var.append(np.unique([ t[ax] for t in T ]))

        cnt = cnt+1

    x = np.reshape(meanScores, newshape=(len(var[1]),len(var[0])))

    xDf = pd.DataFrame(x, columns = var[0], index=var[1])

    print(xDf)

    sb.heatmap(xDf, annot=True, cbar_kws={'label': 'MAE'})

    plt.xlabel(list(axes)[0])

    plt.ylabel(list(axes)[1])

    plt.show()



def EvaluateModel(dataTest, dataTrain, xTrain, yTrain, xTest, yTest, modelX, modelY, tagline):

    modelX.fit(dataTrain, xTrain)

    modelY.fit(dataTrain, yTrain)

    predX = modelX.predict(dataTest)

    predY = modelY.predict(dataTest)

    dists = EucDist(predX, predY, xTest, yTest);

    meanED = np.mean(dists)

    maxED = np.max(dists)

    minED = np.min(dists)

    

    

    print(tagline)

    print("meanED = " + str(meanED) + " m")

    print("maxED = " + str(maxED) + " m")

    print("minED = " + str(minED) + " m")
# read in smartphone data

spData = pd.read_csv('../input/m1SmartPhoneDataWithPosition.csv')

spData = pd.concat([spData, pd.read_csv('../input/m2SmartPhoneDataWithPosition.csv')])

# remove index column coming from R

spData = spData.drop(spData.columns[0], axis=1)

spData.head(3)

# remove missing positions

spDataClean = spData.drop(spData.index[spData['posId'] == -1], axis=0)

spDataClean.reset_index()

spDataClean.head(10)

# correlation check

sb.heatmap(spDataClean.corr(), annot=True)

print(spDataClean.columns)

spDataCleanUnCorr = spDataClean.drop('Z.AxisAgle.Azimuth.', axis=1)

spDataCleanUnCorr.head(3)
# check for potential outliers using Mahalanobis distance

# based on: https://www.machinelearningplus.com/statistics/mahalanobis-distance/

def MahalanobisDist(obs, mu, covMat):

    meanDiff = obs - mu;

    #print(meanDiff.shape)

    invCovMat = np.linalg.inv(covMat)

    #print(invCovMat.shape)

    p1 = np.matmul(invCovMat, meanDiff)

    #print(p1)

    p2 = np.matmul(np.transpose(meanDiff), p1)

    d = np.mean(np.sqrt(p2))

    #print(d)

    return d



onlyMagAngles = spDataClean.iloc[:,0:6]

covMat = onlyMagAngles.cov()

colMeans = onlyMagAngles.mean(axis=0)

print(colMeans)

mDists = []

for i, obs in onlyMagAngles.iterrows():    

    mDists.append(MahalanobisDist(obs, colMeans, covMat))

print(np.array(mDists).shape)

# spDataClean_ just for display purposes

spDataClean_ = spDataClean;

spDataClean_['MahaDists'] = mDists

spDataClean_.sort_values("MahaDists", ascending=False).head(30)

# remove observations that are 60% of the max Mahalanobis distance

maxMDist = np.max(mDists)

spDataCleanOR = spDataClean.drop(spDataClean.index[mDists > (0.6 * maxMDist)], axis=0)

spDataCleanOR.sort_values("MahaDists", ascending=False).head(30)

print(spDataClean.shape)

print(spDataCleanOR.shape)
# do regression here

# use all features for the purposes of classifier tuning

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt







    

data = spDataCleanOR.iloc[:,0:6]

x = spDataCleanOR['x']

y = spDataCleanOR['y']

    

dataTrain, dataTest, xTrain, xTest = train_test_split(data, x, test_size=0.3, random_state=2)

dataTrain, dataTest, yTrain, yTest = train_test_split(data, y, test_size=0.3, random_state=2)



maxIter = 100

numCV = 5



paramsToTest = {'degree': [2,3,4], 'C': [0.001, 0.01, 1]}

gsXResSVR = DoGridSearchTuning(dataTrain, xTrain, numCV, paramsToTest, SVR(kernel="poly", max_iter=maxIter, verbose=False))

gsYResSVR = DoGridSearchTuning(dataTrain, yTrain, numCV, paramsToTest, SVR(kernel="poly", max_iter=maxIter, verbose=False))



paramsToTest = {'min_samples_leaf': [10,20,30], 'n_estimators': [50, 100, 150]}

GBR = GradientBoostingRegressor(random_state=1234)



gsXResGBR = DoGridSearchTuning(dataTrain, xTrain, numCV, paramsToTest, GBR)

gsYResGBR = DoGridSearchTuning(dataTrain, yTrain, numCV, paramsToTest, GBR)



print("Best parameters for X regressor (SVR):")

print(gsXResSVR.best_params_)

print("Best parameters for Y regressor (SVR):")

print(gsYResSVR.best_params_)



print("Best parameters for X regressor (GBR):")

print(gsXResGBR.best_params_)

print("Best parameters for Y regressor (GBR):")

print(gsYResGBR.best_params_)



PlotResultsOfGridSearch(gsXResSVR)

PlotResultsOfGridSearch(gsYResSVR)



PlotResultsOfGridSearch(gsXResGBR)

PlotResultsOfGridSearch(gsYResGBR)



# all features regression (withoutliers removed)

data = spDataCleanOR.iloc[:,0:6]

x = spDataCleanOR['x']

y = spDataCleanOR['y']

    

dataTrain, dataTest, xTrain, xTest = train_test_split(data, x, test_size=0.3, random_state=2)

dataTrain, dataTest, yTrain, yTest = train_test_split(data, y, test_size=0.3, random_state=2)





gsXResSVR.best_params_.update({'max_iter': maxIter, 'kernel': 'poly'})

gsYResSVR.best_params_.update({'max_iter': maxIter, 'kernel': 'poly'})

gsXResGBR.best_params_.update({'random_state': 1234})

gsYResGBR.best_params_.update({'random_state': 1234})



svrX = SVR(**gsXResSVR.best_params_)

svrY = SVR(**gsYResSVR.best_params_)

gbrX = GradientBoostingRegressor(**gsXResGBR.best_params_)

gbrY = GradientBoostingRegressor(**gsYResGBR.best_params_)



EvaluateModel(dataTest, dataTrain, xTrain, yTrain, xTest, yTest, svrX, svrY, '--Metrics for SVR regressor--')

EvaluateModel(dataTest, dataTrain, xTrain, yTrain, xTest, yTest, gbrX, gbrY, '--Metrics for GBR regressor--')

# all features regression (without outliers removed)

data = spDataClean.iloc[:,0:6]

x = spDataClean['x']

y = spDataClean['y']

    

dataTrain, dataTest, xTrain, xTest = train_test_split(data, x, test_size=0.3, random_state=2)

dataTrain, dataTest, yTrain, yTest = train_test_split(data, y, test_size=0.3, random_state=2)





gsXResSVR.best_params_.update({'max_iter': maxIter, 'kernel': 'poly'})

gsYResSVR.best_params_.update({'max_iter': maxIter, 'kernel': 'poly'})

gsXResGBR.best_params_.update({'random_state': 1234})

gsYResGBR.best_params_.update({'random_state': 1234})



svrX = SVR(**gsXResSVR.best_params_)

svrY = SVR(**gsYResSVR.best_params_)

gbrX = GradientBoostingRegressor(**gsXResGBR.best_params_)

gbrY = GradientBoostingRegressor(**gsYResGBR.best_params_)



EvaluateModel(dataTest, dataTrain, xTrain, yTrain, xTest, yTest, svrX, svrY, '--Metrics for SVR regressor--')

EvaluateModel(dataTest, dataTrain, xTrain, yTrain, xTest, yTest, gbrX, gbrY, '--Metrics for GBR regressor--')
# all features regression (withoutliers removed)

data = spDataCleanUnCorr.iloc[:,0:6]

x = spDataCleanUnCorr['x']

y = spDataCleanUnCorr['y']

    

dataTrain, dataTest, xTrain, xTest = train_test_split(data, x, test_size=0.3, random_state=2)

dataTrain, dataTest, yTrain, yTest = train_test_split(data, y, test_size=0.3, random_state=2)





gsXResSVR.best_params_.update({'max_iter': maxIter, 'kernel': 'poly'})

gsYResSVR.best_params_.update({'max_iter': maxIter, 'kernel': 'poly'})

gsXResGBR.best_params_.update({'random_state': 1234})

gsYResGBR.best_params_.update({'random_state': 1234})



svrX = SVR(**gsXResSVR.best_params_)

svrY = SVR(**gsYResSVR.best_params_)

gbrX = GradientBoostingRegressor(**gsXResGBR.best_params_)

gbrY = GradientBoostingRegressor(**gsYResGBR.best_params_)



EvaluateModel(dataTest, dataTrain, xTrain, yTrain, xTest, yTest, svrX, svrY, '--Metrics for SVR regressor--')

EvaluateModel(dataTest, dataTrain, xTrain, yTrain, xTest, yTest, gbrX, gbrY, '--Metrics for GBR regressor--')