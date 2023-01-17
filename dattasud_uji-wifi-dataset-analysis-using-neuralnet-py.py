# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import seaborn as sb

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



def EucDist(x1, y1, x2, y2):

    return np.sqrt( np.power((x1-x2),2) + np.power((y1-y2), 2) )



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
# Any results you write to the current directory are saved as output.

ds_ = pd.read_csv('../input/UJIData.csv')

ds = ds_.drop(ds_.columns[0], axis=1)
ujidataset = ds.iloc[:, 1:520]

ujidataset = ujidataset.replace(to_replace=100, value=-200)



#check corr

corr_matrix = ujidataset.corr().abs()

#sb.heatmap(ujidataset.corr(), annot=True)

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]

#print("Dropped columns based on high corr > 0.80:")

#print(to_drop)

# Drop features 

ujidataset = ujidataset.drop(ujidataset[to_drop], axis=1)

#print("Columns used by NN model:")

#for col in ujidataset.columns:

#    print(col)

ujidatasetLat = ds['LATITUDE']

#print(ujidatasetLat)

ujidatasetLong = ds['LONGITUDE']

#print(ujidatasetLong)



xTrain, xTest, latTrainDS, latTestDS = train_test_split(ujidataset, ujidatasetLat, test_size=0.3, random_state=2)

yTrain, yTest, lonTrainDS, lonTestDS = train_test_split(ujidataset, ujidatasetLong, test_size=0.3, random_state=2)



nnModLat = MLPRegressor(hidden_layer_sizes=(5,3),

                     activation='tanh',

                     solver='lbfgs',

                     verbose=True)



nnModLat.fit(xTrain, latTrainDS)

predLat = nnModLat.predict(xTest)

print(predLat)

nnModLong = MLPRegressor(hidden_layer_sizes=(5,3),

                     activation='tanh',

                     solver='lbfgs',

                     verbose=True)



nnModLong.fit(yTrain, lonTrainDS)

predLong = nnModLong.predict(yTest)

print("MAE Latitude = " + str(np.mean(np.absolute(predLat - latTestDS))) + " m")

print("MAE Longitude = " + str(np.mean(np.absolute(predLong - lonTestDS))) + " m")



EvaluateModel(xTest, xTrain, latTrainDS, lonTrainDS, latTestDS, lonTestDS, nnModLat, nnModLong, "--Metrics for Neural Network--")
# do a grid search for the neural network

numCV = 5

print(xTrain.columns)

paramsToTest = {'hidden_layer_sizes': [(3,),(5,3),(5,3,3)], 'activation': ['tanh', 'relu']}

NNResLat = DoGridSearchTuning(xTrain, latTrainDS, numCV, paramsToTest, MLPRegressor(solver='lbfgs', verbose=True))

NNResLon = DoGridSearchTuning(xTrain, lonTrainDS, numCV, paramsToTest, MLPRegressor(solver='lbfgs', verbose=True))

PlotResultsOfGridSearch(NNResLat)

PlotResultsOfGridSearch(NNResLon)

print("Best parameters for latitude regressor (NN):")

print(NNResLat.best_params_)

print("Best parameters for longitude regressor (NN):")

print(NNResLon.best_params_)

nnModLat = MLPRegressor(hidden_layer_sizes=(3,),

                     activation='tanh',

                     solver='lbfgs',

                     verbose=True)



nnModLat.fit(xTrain, latTrainDS)

predLat = nnModLat.predict(xTest)

#print(predLat)

nnModLong = MLPRegressor(hidden_layer_sizes=(3,),

                     activation='relu',

                     solver='lbfgs',

                     verbose=True)



nnModLong.fit(yTrain, lonTrainDS)

predLong = nnModLong.predict(yTest)

print("MAE Latitude = " + str(np.mean(np.absolute(predLat - latTestDS))) + " m")

print("MAE Longitude = " + str(np.mean(np.absolute(predLong - lonTestDS))) + " m")



EvaluateModel(xTest, xTrain, latTrainDS, lonTrainDS, latTestDS, lonTestDS, nnModLat, nnModLong, "--Metrics for Neural Network--")