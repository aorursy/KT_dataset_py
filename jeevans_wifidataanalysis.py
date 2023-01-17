# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR

import matplotlib.pyplot as plt

import seaborn as sb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#function definitions

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



def RemoveOutliersByMD(df):

    onlyMags = df.iloc[:,1:4]

    #print(onlyMags)

    covMat = onlyMags.cov()

    colMeans = onlyMags.mean(axis=0)

    print(colMeans)

    mDists = []

    for i, obs in onlyMags.iterrows():    

        mDists.append(MahalanobisDist(obs, colMeans, covMat))

    #print(np.array(mDists).shape)

    # spDataClean_ just for display purposes

    # spDataClean_ = spDataClean;

    # spDataClean_['MahaDists'] = mDists

    # spDataClean_.sort_values("MahaDists", ascending=False).head(30)

    # remove observations that are 60% of the max Mahalanobis distance

    maxMDist = np.max(mDists)

    dfOR = df.drop(df.index[mDists > (0.6 * maxMDist)], axis=0)

    #spDataCleanOR.sort_values("MahaDists", ascending=False).head(30)

    return dfOR;



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

    x = np.reshape(meanScores, newshape=(len(var[0]),len(var[1])))

    xDf = pd.DataFrame(x, columns = var[1], index=var[0])

    print(xDf)

    plt.figure(figsize=(9,6))

    sb.heatmap(xDf, annot=True, cbar_kws={'label': 'MAE'}, fmt='.6g')

    plt.xlabel(list(axes)[1])

    plt.ylabel(list(axes)[0])

    plt.show()



def EvaluateModel(dataTrain, dataTest, xTrain, yTrain, xTest, yTest, modelX, modelY, tagline):

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

    plt.figure(figsize=(9,6))

    sb.distplot(dists, hist=True, kde=False, 

             bins=100, color = 'blue',

             hist_kws={'edgecolor':'black'})

    

    plt.title(tagline)

    plt.ylabel('Frequency of Error')

    plt.xlabel('Euclidean Distance (m)')

    

def EvaluateModelNN(dataTrain, dataTest, xyTrain, xyTest, model, tagline):

    model.fit(dataTrain, xyTrain)

    pred = model.predict(dataTest)

    dists = EucDist(pred[:,0], pred[:,1], xyTest[:,0], xyTest[:,1]);

    meanED = np.mean(dists)

    maxED = np.max(dists)

    minED = np.min(dists)

    print(tagline)

    print("meanED = " + str(meanED) + " m")

    print("maxED = " + str(maxED) + " m")

    print("minED = " + str(minED) + " m")

    

def DoModelAnalysis(Tr, Te, yTr, yTe, scaler, stdParams, expParams, numFolds, model, modelName, dataName):

    # do scaling

    Tr = scaler.transform(Tr)

    Te = scaler.transform(Te)

    

    # do grid search

    gsX = DoGridSearchTuning(Tr, yTr[:,0], numCV, expParams, model(**stdParams))

    gsY = DoGridSearchTuning(Tr, yTr[:,1], numCV, expParams, model(**stdParams))

    print('Best parameters for X regressor ' + str(modelName) + ' on ' + str(dataName) + ":")

    print(gsX.best_params_)

    PlotResultsOfGridSearch(gsX)

    

    print('Best parameters for Y regressor ' + str(modelName) + ' on ' + str(dataName) + ":")

    print(gsY.best_params_)

    PlotResultsOfGridSearch(gsY)

  

    gsX.best_params_.update(stdParams)

    gsY.best_params_.update(stdParams)

    #modelXEval = model.set_params(**gsX.best_params_)

    modelXEval = model(**gsX.best_params_)

    

    #modelXEval = MLPRegressor(**gsX.best_params_)

    #print(modelXEval)

    #modelYEval = model.set_params(**gsY.best_params_)

    modelYEval = model(**gsY.best_params_)

    

    #modelYEval = MLPRegressor(**gsY.best_params_)

    #print(modelYEval)

    

#     if modelName == 'Neural Network':

#         modelXEval = MLPRegressor(**gsX.best_params_)

#         modelXEval = MLPRegressor(**gsX.best_params_)

        

    

    modelXEval.fit(Tr, yTr[:,0])

    modelYEval.fit(Tr, yTr[:,1])

    

    EvaluateModel(Tr, Te, yTr[:,0], yTr[:,1], yTe[:,0], yTe[:,1], modelXEval, modelYEval, '-- Metrics for ' + str(modelName) + ' on ' + str(dataName) + ' --')



    
# read data and drop 1st column

ds_ = pd.read_csv('../input/UJIData.csv')

ds = ds_.drop(ds_.columns[0], axis=1)

ds.head(10)
# check correlation and remove manually highly correlated columns

wifiDf = ds.iloc[:, 1:521]

wifiDf.replace(to_replace=100, value=-200)

lats = ds['LATITUDE']

#print(lats)

lons = ds['LONGITUDE']



print(wifiDf.shape)

plt.figure(figsize=(9,6))

sb.heatmap(wifiDf.corr())



# calculate correlation of WAP signal strength measurements

c = wifiDf.corr()

# mark points in correlation matrix were correlation exceeds 80%

cLogic = c > 0.8

# perform column sum of all correlations that exceed 80%

cLogicSum = np.sum(cLogic, axis=0)

print(np.sort(cLogicSum))

# mark predictor to remove as a column whose correlation sum exceeds 3.

varsToRemove = np.where(cLogicSum >= 3)

print(varsToRemove)

# remove highly correlated preductors from dataset

wifiDfUncorr = wifiDf.drop(wifiDf.columns[varsToRemove], axis=1)
# do PCA and create PCA dataset

from sklearn.decomposition import PCA

pcaWifi = PCA()



pcaWifi.fit(wifiDf)

totalWifiVar = np.sum(np.var(wifiDf))

#print(pcaWifi.components_)

plt.xlabel('Principal components')

plt.ylabel('Percentage variance explained')

plt.plot(pcaWifi.explained_variance_/totalWifiVar)

# scree plot here

plt.figure()

plt.xlabel('Principal components')

plt.ylabel('Cumulative percentage variance explained')

plt.plot(np.cumsum(pcaWifi.explained_variance_)/totalWifiVar)



#keep only principal components that account for 80% of the variance

numComponents = np.sum(( (np.cumsum(pcaWifi.explained_variance_)/totalWifiVar) <= 0.9 ))

print(numComponents)



#do fitting to 1st numComponents PCA scores (optimized parameters)

loadings = pcaWifi.components_

print(loadings.shape)

#print(scores)



scores = np.matmul(np.matrix(wifiDf), loadings)

print(scores)

scoresToUse = scores[:, 0:numComponents]
# create data splits (70-30)

# all vars

xTrainAll, xTestAll, latTrainAll, latTestAll = train_test_split(wifiDf, lats, test_size=0.3, random_state=2)

xTrainAll, xTestAll, lonTrainAll, lonTestAll = train_test_split(wifiDf, lons, test_size=0.3, random_state=2)

latLonTrainAll = np.stack((latTrainAll, lonTrainAll), axis=1)

latLonTestAll = np.stack((latTestAll, lonTestAll), axis=1)



# uncorr data

xTrainUC, xTestUC, latTrainUC, latTestUC = train_test_split(wifiDfUncorr, lats, test_size=0.3, random_state=2)

xTrainUC, xTestUC, lonTrainUC, lonTestUC = train_test_split(wifiDfUncorr, lons, test_size=0.3, random_state=2)

latLonTrainUC = np.stack((latTrainUC, lonTrainUC), axis=1)

latLonTestUC = np.stack((latTestUC, lonTestUC), axis=1)



# PCA data

xTrainPCA, xTestPCA, latTrainPCA, latTestPCA = train_test_split(scoresToUse, lats, test_size=0.3, random_state=2)

xTrainPCA, xTestPCA, lonTrainPCA, lonTestPCA = train_test_split(scoresToUse, lons, test_size=0.3, random_state=2)

latLonTrainPCA = np.stack((latTrainPCA, lonTrainPCA), axis=1)

latLonTestPCA = np.stack((latTestPCA, lonTestPCA), axis=1)
# Do model analysis for neural network (all vars)

numCV = 5

stdParamsNN = {'solver': 'lbfgs', 'random_state': 4}

paramsToTestNN = {'hidden_layer_sizes': [(3,),(5,3),(5,3,3)], 'activation': ['logistic', 'relu','tanh']}



scalerAll = StandardScaler()

scalerAll.fit(xTrainAll)

DoModelAnalysis(xTrainAll, xTestAll, latLonTrainAll, latLonTestAll, scalerAll, 

               stdParamsNN, paramsToTestNN, numCV, MLPRegressor, 'Neural Network', 'All Variables')



scalerUC = StandardScaler()

scalerUC.fit(xTrainUC)

DoModelAnalysis(xTrainUC, xTestUC, latLonTrainUC, latLonTestUC, scalerUC, 

               stdParamsNN, paramsToTestNN, numCV, MLPRegressor, 'Neural Network', 'Uncorrelated Variables')



scalerPCA = StandardScaler()

scalerPCA.fit(xTrainPCA)

DoModelAnalysis(xTrainPCA, xTestPCA, latLonTrainPCA, latLonTestPCA, scalerPCA, 

               stdParamsNN, paramsToTestNN, numCV, MLPRegressor, 'Neural Network', 'PCA Scores')

# Do model analysis for Random Forest

numCV = 5

stdParamsRF = {'max_features': 'sqrt', 'n_jobs': -1, 'random_state': 4}

paramsToTestRF = {'n_estimators': [150, 300, 450], 'min_samples_leaf': [5, 10, 15, 20]}



stdParams = stdParamsRF

paramsToTest = paramsToTestRF

modelName = 'Random Forest'

model = RandomForestRegressor



scalerAll = StandardScaler()

scalerAll.fit(xTrainAll)

DoModelAnalysis(xTrainAll, xTestAll, latLonTrainAll, latLonTestAll, scalerAll, 

               stdParams, paramsToTest, numCV, model, modelName, 'All Variables')



scalerUC = StandardScaler()

scalerUC.fit(xTrainUC)

DoModelAnalysis(xTrainUC, xTestUC, latLonTrainUC, latLonTestUC, scalerUC,

               stdParams, paramsToTest, numCV, model, modelName, 'Uncorrelated Variables')



scalerPCA = StandardScaler()

scalerPCA.fit(xTrainPCA)

DoModelAnalysis(xTrainPCA, xTestPCA, latLonTrainPCA, latLonTestPCA, scalerPCA,

               stdParams, paramsToTest, numCV, model, modelName, 'PCA Scores')

# Do model analysis for Random Forest

numCV = 5

stdParamsGBR = {'random_state': 4}

paramsToTestGBR = {'n_estimators': [150, 300, 450], 'min_samples_leaf': [5, 10, 15]}



stdParams = stdParamsGBR

paramsToTest = paramsToTestGBR

modelName = 'Gradient Boosting Trees'

model = GradientBoostingRegressor



scalerAll = StandardScaler()

scalerAll.fit(xTrainAll)

DoModelAnalysis(xTrainAll, xTestAll, latLonTrainAll, latLonTestAll, scalerAll, 

               stdParams, paramsToTest, numCV, model, modelName, 'All Variables')



scalerUC = StandardScaler()

scalerUC.fit(xTrainUC)

DoModelAnalysis(xTrainUC, xTestUC, latLonTrainUC, latLonTestUC, scalerUC,

               stdParams, paramsToTest, numCV, model, modelName, 'Uncorrelated Variables')



scalerPCA = StandardScaler()

scalerPCA.fit(xTrainPCA)

DoModelAnalysis(xTrainPCA, xTestPCA, latLonTrainPCA, latLonTestPCA, scalerPCA,

               stdParams, paramsToTest, numCV, model, modelName, 'PCA Scores')
# Do model analysis for Random Forest

numCV = 5

stdParamsSVR = {'max_iter': 1000, 'kernel': 'poly'}

paramsToTestSVR = {'degree': [2,3,4], 'C': [0.001, 0.01, 1]}



stdParams = stdParamsSVR

paramsToTest = paramsToTestSVR

modelName = 'Support Vector Regressor'

model = SVR



scalerAll = StandardScaler()

scalerAll.fit(xTrainAll)

DoModelAnalysis(xTrainAll, xTestAll, latLonTrainAll, latLonTestAll, scalerAll, 

               stdParams, paramsToTest, numCV, model, modelName, 'All Variables')



scalerUC = StandardScaler()

scalerUC.fit(xTrainUC)

DoModelAnalysis(xTrainUC, xTestUC, latLonTrainUC, latLonTestUC, scalerUC,

               stdParams, paramsToTest, numCV, model, modelName, 'Uncorrelated Variables')



scalerPCA = StandardScaler()

scalerPCA.fit(xTrainPCA)

DoModelAnalysis(xTrainPCA, xTestPCA, latLonTrainPCA, latLonTestPCA, scalerPCA,

               stdParams, paramsToTest, numCV, model, modelName, 'PCA Scores')