import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sb

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split



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



ds = pd.read_csv('../input/smartphone.csv')

ds = ds.drop(ds.columns[0], axis=1)

ds = ds.drop(ds.index[ds['posId'] == -1], axis=0)

smartphoneds = ds.drop(ds.columns[0], axis=1)

smartphoneds.head(5)

smartphoneds = smartphoneds.iloc[:, 0:6]



smartphoneds.head(3)
corr_matrix = smartphoneds.corr().abs()

sb.heatmap(smartphoneds.corr(), annot=True)

for col in smartphoneds.columns:

    print(col)

#smartphoneds = smartphoneds.drop('Z.AxisAgle.Azimuth.', axis=1)

smartphoneds.head(3)
smartphonedsX = ds['x']

smartphonedsY = ds['y']



smartphonedsX = ds['x']

smartphonedsY = ds['y']

#print(smartwatchds.columns)

smartphoneds_ = smartphoneds.iloc[:,0:6]

print(smartphoneds_.columns)

    

xTrain, xTest, xTrainDS, xTestDS = train_test_split(smartphoneds_, smartphonedsX, test_size=0.3, random_state=2)

yTrain, yTest, yTrainDS, yTestDS = train_test_split(smartphoneds_, smartphonedsY, test_size=0.3, random_state=2)



nnModX = MLPRegressor(hidden_layer_sizes=(5,3),

                     activation='tanh',

                     solver='lbfgs',

                     verbose=True)



nnModX.fit(xTrain, xTrainDS)

predX = nnModX.predict(xTest)

print(predX)
nnModY = MLPRegressor(hidden_layer_sizes=(5,3),

                     activation='tanh',

                     solver='lbfgs',

                     verbose=True)



nnModY.fit(yTrain, yTrainDS)

predY = nnModY.predict(yTest)

print(predY)
dists = EucDist(predX, predY, xTestDS, yTestDS)

meanED = np.mean(dists)

maxED = np.max(dists)

minED = np.min(dists)



print("meanED = " + str(meanED) + " m")

print("maxED = " + str(maxED) + " m")

print("minED = " + str(minED) + " m")
# quick grid search

numCV = 5

paramsToTest = {'hidden_layer_sizes': [(3,),(5,3),(5,3,3)], 'activation': ['tanh', 'relu']}

NNResX = DoGridSearchTuning(xTrain, xTrainDS, numCV, paramsToTest, MLPRegressor(solver='lbfgs', verbose=True))

NNResY = DoGridSearchTuning(xTrain, yTrainDS, numCV, paramsToTest, MLPRegressor(solver='lbfgs', verbose=True))

PlotResultsOfGridSearch(NNResX)

PlotResultsOfGridSearch(NNResY)

print("Best parameters for X regressor (NN):")

print(NNResX.best_params_)

print("Best parameters for Y regressor (NN):")

print(NNResY.best_params_)
# do regression with optimal parameters here (with Azimuth)

smartphonedsX = ds['x']

smartphonedsY = ds['y']

#print(smartwatchds.columns)

smartphoneds_ = smartphoneds.iloc[:,0:6]

print(smartphoneds_.columns)

    

xTrain, xTest, xTrainDS, xTestDS = train_test_split(smartphoneds_, smartphonedsX, test_size=0.3, random_state=2)

yTrain, yTest, yTrainDS, yTestDS = train_test_split(smartphoneds_, smartphonedsY, test_size=0.3, random_state=2)



nnModY.fit(yTrain, yTrainDS)

predY = nnModY.predict(yTest)

print(predY)



NNResX.best_params_.update({'solver': 'lbfgs'})

NNResY.best_params_.update({'solver': 'lbfgs'})



nnModX = MLPRegressor(**NNResX.best_params_)

nnModY = MLPRegressor(**NNResY.best_params_)

nnModX.fit(xTrain, xTrainDS)

nnModY.fit(yTrain, yTrainDS)



EvaluateModel(xTest, xTrain, xTrainDS, yTrainDS, xTestDS, yTestDS, nnModX, nnModY, '--Metrics for Neural Network--')

# do regression with optimal parameters here (without Azimuth)

smartphoneds_ = ds.drop('Z.AxisAgle.Azimuth.', axis=1)

smartphonedsX = ds['x']

smartphonedsY = ds['y']

#print(smartwatchds.columns)

smartphoneds_ = smartphoneds_.iloc[:,0:6]

print(smartphoneds_.columns)

    

xTrain, xTest, xTrainDS, xTestDS = train_test_split(smartphoneds_, smartphonedsX, test_size=0.3, random_state=2)

yTrain, yTest, yTrainDS, yTestDS = train_test_split(smartphoneds_, smartphonedsY, test_size=0.3, random_state=2)



nnModY.fit(yTrain, yTrainDS)

predY = nnModY.predict(yTest)

print(predY)



NNResX.best_params_.update({'solver': 'lbfgs'})

NNResY.best_params_.update({'solver': 'lbfgs'})



nnModX = MLPRegressor(**NNResX.best_params_)

nnModY = MLPRegressor(**NNResY.best_params_)

nnModX.fit(xTrain, xTrainDS)

nnModY.fit(yTrain, yTrainDS)



EvaluateModel(xTest, xTrain, xTrainDS, yTrainDS, xTestDS, yTestDS, nnModX, nnModY, '--Metrics for Neural Network--')
