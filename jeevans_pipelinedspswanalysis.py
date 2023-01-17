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



    
# smartphone data

spData = pd.read_csv('../input/m1SmartPhoneDataWithPosition.csv')

spData = pd.concat([spData, pd.read_csv('../input/m2SmartPhoneDataWithPosition.csv')])

spData = spData.drop(spData.columns[0], axis=1)

#spData = spData.drop(spData.columns[0], axis=1)

spData = spData.reset_index()

#spData.head(10)

spDataClean = spData.drop(spData.index[spData['posId'] == -1])

print(spDataClean.shape)

spDataClean.head(10)



# smartwatch data

swData = pd.read_csv('../input/m1SmartWatchDataWithPosition.csv')

swData = pd.concat([swData, pd.read_csv('../input/m2SmartWatchDataWithPosition.csv')])

swData = swData.drop(swData.columns[0], axis=1)

#spData = spData.drop(spData.columns[0], axis=1)

swData = swData.reset_index()

#spData.head(10)

swDataClean = swData.drop(swData.index[swData['posId'] == -1])

print(swDataClean.shape)

swDataClean.head(10)
# do outlier removal

swDataCleanOR = RemoveOutliersByMD(swDataClean)

spDataCleanOR = RemoveOutliersByMD(spDataClean)

#swDataCleanOR = swDataClean

#spDataCleanOR = spDataClean

print(swDataCleanOR.shape)

print(spDataCleanOR.shape)
# do data splitting

data = spDataCleanOR

x = data['x']

y = data['y']

data = data.iloc[:,[1,2,3,5,6]]

print(data.columns)



spTr, spTe, spOutXTr, spOutXTe = train_test_split(data, x, test_size=0.3, random_state=2)

spTr, spTe, spOutYTr, spOutYTe = train_test_split(data, y, test_size=0.3, random_state=2)

spScaler = StandardScaler()

spScaler.fit(spTr)

#spTr = spScaler.transform(spTr)



data = swDataCleanOR

x = data['x']

y = data['y']

data = data.iloc[:,[1,2,3,4,5]]

print(data.columns)



swTr, swTe, swOutXTr, swOutXTe = train_test_split(data, x, test_size=0.3, random_state=2)

swTr, swTe, swOutYTr, swOutYTe = train_test_split(data, y, test_size=0.3, random_state=2)

swScaler = StandardScaler()

swScaler.fit(swTr)

#swTr = swScaler.transform(swTr)
# do grid search for neural net (with x and y simultaneously)

numCV=5

stdParamsNN = {'solver': 'lbfgs', 'random_state': 4}

paramsToTestNN = {'hidden_layer_sizes': [(3,),(5,3),(5,3,3)], 'activation': ['logistic', 'relu','tanh']}

xySW = np.stack((swOutXTr, swOutYTr), axis=1)

# print(xySW.shape)

# nnSWGSRes = DoGridSearchTuning(swTr, xySW, numCV, paramsToTest, MLPRegressor(**stdParams))

xySP = np.stack((spOutXTr, spOutYTr), axis=1)

# print(xySP[:,1])

# print(xySP.shape)

# nnSPGSRes = DoGridSearchTuning(spTr, xySP, numCV, paramsToTest, MLPRegressor(**stdParams))



# print(nnSWGSRes.best_params_)

# PlotResultsOfGridSearch(nnSWGSRes)



# print(nnSPGSRes.best_params_)

# PlotResultsOfGridSearch(nnSPGSRes)



actuXYSP = np.stack((spOutXTe, spOutYTe), axis=1)

actuXYSW = np.stack((swOutXTe, swOutYTe), axis=1)



# Do model analysis for neural network

# DoModelAnalysis(spTr, spTe, xySP, actuXYSP, spScaler, stdParamsNN, paramsToTestNN, numCV, 

#                 MLPRegressor(**stdParamsNN), 'Neural Network', 'Smartphone Data')



DoModelAnalysis(spTr, spTe, xySP, actuXYSP, spScaler, stdParamsNN, paramsToTestNN, numCV, 

                MLPRegressor, 'Neural Network', 'Smartphone Data')





DoModelAnalysis(swTr, swTe, xySW, actuXYSW, swScaler, stdParamsNN, paramsToTestNN, numCV, 

                MLPRegressor, 'Neural Network', 'Smartwatch Data')





# # do fitting and evaluation for NN

# spTe = spScaler.transform(spTe)

# swTe = swScaler.transform(swTe)



# nnSWGSRes.best_params_.update(stdParams)

# nnSPGSRes.best_params_.update(stdParams)

# nnSP = MLPRegressor(**nnSPGSRes.best_params_)

# nnSW = MLPRegressor(**nnSWGSRes.best_params_)

# # nnSP.fit(spTr, xySP)

# # nnSW.fit(swTr, xySW)



# actuXYSW = np.stack((swOutXTe, swOutYTe), axis=1)

# actuXYSP = np.stack((spOutXTe, spOutYTe), axis=1)



# EvaluateModel(swTr, swTe, xySW, actuXYSW, nnSW, '-- Metrics for NN smartwatch data --')

# EvaluateModel(spTr, spTe, xySP, actuXYSP, nnSP, '-- Metrics for NN smartphone data --')

stdParamsRF = {'max_features': 'sqrt', 'n_jobs': -1, 'random_state': 4}

paramsToTestRF = {'n_estimators': [150, 300, 450], 'min_samples_leaf': [5, 10, 15, 20]}



DoModelAnalysis(spTr, spTe, xySP, actuXYSP, spScaler, stdParamsRF, paramsToTestRF, numCV, 

                RandomForestRegressor, 'Random Forest', 'Smartphone Data')



DoModelAnalysis(swTr, swTe, xySW, actuXYSW, swScaler, stdParamsRF, paramsToTestRF, numCV, 

                RandomForestRegressor, 'Random Forest', 'Smartwatch Data')
stdParamsGBR = {'random_state': 4}

paramsToTestGBR = {'n_estimators': [150, 300, 450], 'min_samples_leaf': [5, 10, 15, 20]}



DoModelAnalysis(spTr, spTe, xySP, actuXYSP, spScaler, stdParamsGBR, paramsToTestGBR, numCV, 

                GradientBoostingRegressor, 'Boosting Trees', 'Smartphone Data')



DoModelAnalysis(swTr, swTe, xySW, actuXYSW, swScaler, stdParamsGBR, paramsToTestGBR, numCV, 

                GradientBoostingRegressor, 'Boosting Trees', 'Smartwatch Data')
stdParamsSVR = {'max_iter': 1000, 'kernel': 'poly'}

paramsToTestSVR = {'degree': [2,3,4], 'C': [0.001, 0.01, 1]}



DoModelAnalysis(spTr, spTe, xySP, actuXYSP, spScaler, stdParamsSVR, paramsToTestSVR, numCV, 

                SVR, 'Support Vector Machine', 'Smartphone Data')



DoModelAnalysis(swTr, swTe, xySW, actuXYSW, swScaler, stdParamsSVR, paramsToTestSVR, numCV, 

                SVR, 'Support Vector Machine', 'Smartwatch Data')