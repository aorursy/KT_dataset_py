#Libraries to use 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

import seaborn as sb

import matplotlib.pyplot as plt



# Function definitions

def EucDist(x1, y1, x2, y2):

    return np.sqrt( np.power((x1-x2),2) + np.power((y1-y2), 2) )



def DoGridSearchTuning(X, y, numFolds, params, model):

    gsRes = GridSearchCV(model, param_grid=params, cv=numFolds, n_jobs=-1, scoring='neg_mean_absolute_error')

    gsRes.fit(X,y)

    return gsRes



def PlotResultsOfGridSearch(gsResult):

    meanScores = np.absolute(gsResult.cv_results_['mean_test_score'])

    print(meanScores)

    T = gsResult.cv_results_['params']

    axes = T[0].keys()

    cnt = 0

    var = []

    for ax in axes:

        var.append(np.unique([ t[ax] for t in T ]))

        cnt = cnt+1

    x = np.reshape(meanScores, newshape=(len(var[1]),len(var[0])))

    xDf = pd.DataFrame(x, columns = var[0], index=var[1])

    #print(x)

    sb.heatmap(xDf, annot=True, cbar_kws={'label': 'MAE'})

    plt.xlabel(list(axes)[0])

    plt.ylabel(list(axes)[1])

    plt.show()

    

# def EvaluateModel(model, lat, yTest):

#     dists = EucDist(latPred, lonPred, latTest, lonTest)

    

    
# read data and drop 1st column

ds_ = pd.read_csv('../input/UJIData.csv')

ds = ds_.drop(ds_.columns[0], axis=1)

ds.head(10)

# do fitting, one for lat and one for lon

# use all features for the purposes of classifier tuning

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split



wifiDf = ds.iloc[:, 1:521]

wifiDf.replace(to_replace=100, value=-200)

#print(wifiDf)

lats = ds['LATITUDE']

#print(lats)

lons = ds['LONGITUDE']

#lons



xTrain, xTest, latTrain, latTest = train_test_split(wifiDf, lats, test_size=0.3, random_state=2)

xTrain, xTest, lonTrain, lonTest = train_test_split(wifiDf, lons, test_size=0.3, random_state=2)



maxIter = 100

numCV = 5



paramsToTest = {'degree': [2,3,4], 'C': [0.001, 0.01, 1]}

gsLatResSVR = DoGridSearchTuning(xTrain, latTrain, numCV, paramsToTest, SVR(kernel="poly", max_iter=maxIter, verbose=False))

gsLonResSVR = DoGridSearchTuning(xTrain, lonTrain, numCV, paramsToTest, SVR(kernel="poly", max_iter=maxIter, verbose=False))



paramsToTest = {'min_samples_leaf': [10,20,30], 'n_estimators': [50, 100, 150]}

GBR = GradientBoostingRegressor(random_state = 1234)



gsLatResGBR = DoGridSearchTuning(xTrain, latTrain, numCV, paramsToTest, GBR)

gsLonResGBR = DoGridSearchTuning(xTrain, lonTrain, numCV, paramsToTest, GBR)



print("Best parameters for latitude regressor (SVR):")

print(gsLatResSVR.best_params_)

print("Best parameters for longitude regressor (SVR):")

print(gsLonResSVR.best_params_)



print("Best parameters for latitude regressor (GBR):")

print(gsLatResGBR.best_params_)

print("Best parameters for longitude regressor (GBR):")

print(gsLonResGBR.best_params_)



PlotResultsOfGridSearch(gsLatResSVR)

PlotResultsOfGridSearch(gsLonResSVR)



PlotResultsOfGridSearch(gsLatResGBR)

PlotResultsOfGridSearch(gsLonResGBR)



# do feature reduction by correlation removal



print(wifiDf.shape)



sb.heatmap(wifiDf.corr())





c = wifiDf.corr()

cLogic = c > 0.8

cLogicSum = np.sum(cLogic, axis=0)

print(np.sort(cLogicSum))

varsToRemove = np.where(cLogicSum >= 3)

print(varsToRemove)



wifiDfUncorr = wifiDf.drop(wifiDf.columns[varsToRemove], axis=1)



print(wifiDfUncorr.columns)



xTrain, xTest, latTrain, latTest = train_test_split(wifiDfUncorr, lats, test_size=0.3, random_state=2)

xTrain, xTest, lonTrain, lonTest = train_test_split(wifiDfUncorr, lons, test_size=0.3, random_state=2)



svrLatUncorr = SVR(kernel="poly",

             degree=gsLatResSVR.best_params_['degree'],

             C=gsLatResSVR.best_params_['C'],

             verbose=False,

             max_iter=maxIter)

svrLonUncorr = SVR(kernel="poly",

             degree=gsLonResSVR.best_params_['degree'],

             C=gsLonResSVR.best_params_['C'],

             verbose=False,

             max_iter=maxIter)



gbrLatUncorr = GradientBoostingRegressor(random_state = 1234, 

                   n_estimators = gsLatResGBR.best_params_['n_estimators'],

                   min_samples_leaf = gsLatResGBR.best_params_['min_samples_leaf'])



gbrLonUncorr = GradientBoostingRegressor(random_state = 1234, 

                   n_estimators = gsLatResGBR.best_params_['n_estimators'],

                   min_samples_leaf = gsLatResGBR.best_params_['min_samples_leaf'])



svrLatUncorr.fit(xTrain, latTrain)

svrLonUncorr.fit(xTrain, lonTrain)



gbrLatUncorr.fit(xTrain, latTrain)

gbrLonUncorr.fit(xTrain, lonTrain)





latPred = svrLatUncorr.predict(xTest)

lonPred = svrLonUncorr.predict(xTest)



latPredGBR = gbrLatUncorr.predict(xTest)

lonPredGBR = gbrLonUncorr.predict(xTest)



dists = EucDist(latPred, lonPred, latTest, lonTest)

meanED = np.mean(dists)

maxED = np.max(dists)

minED = np.min(dists)

print("--Metrics for SVR--")

print("meanED = " + str(meanED) + " m")

print("maxED = " + str(maxED) + " m")

print("minED = " + str(minED) + " m")



distsGBR = EucDist(latPredGBR, lonPredGBR, latTest, lonTest)

meanED = np.mean(distsGBR)

maxED = np.max(distsGBR)

minED = np.min(distsGBR)

print("--Metrics for GBR--")

print("meanED = " + str(meanED) + " m")

print("maxED = " + str(maxED) + " m")

print("minED = " + str(minED) + " m")



# do feature reduction here by PCA

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



xTrain, xTest, latTrain, latTest = train_test_split(scoresToUse, lats, test_size=0.3, random_state=2)

xTrain, xTest, lonTrain, lonTest = train_test_split(scoresToUse, lons, test_size=0.3, random_state=2)



svrLatPCA = SVR(kernel="poly",

             degree=gsLatResSVR.best_params_['degree'],

             C=gsLatResSVR.best_params_['C'],

             verbose=False,

             max_iter=maxIter)

svrLonPCA = SVR(kernel="poly",

             degree=gsLonResSVR.best_params_['degree'],

             C=gsLonResSVR.best_params_['C'],

             verbose=False,

             max_iter=maxIter)



gbrLatPCA = GradientBoostingRegressor(random_state = 1234, 

                   n_estimators = gsLatResGBR.best_params_['n_estimators'],

                   min_samples_leaf = gsLatResGBR.best_params_['min_samples_leaf'])



gbrLonPCA = GradientBoostingRegressor(random_state = 1234, 

                   n_estimators = gsLatResGBR.best_params_['n_estimators'],

                   min_samples_leaf = gsLatResGBR.best_params_['min_samples_leaf'])



svrLatPCA.fit(xTrain, latTrain)

svrLonPCA.fit(xTrain, lonTrain)

latPred = svrLatPCA.predict(xTest)

lonPred = svrLonPCA.predict(xTest)



gbrLatPCA.fit(xTrain, latTrain)

gbrLonPCA.fit(xTrain, lonTrain)



latPredGBR = gbrLatPCA.predict(xTest)

lonPredGBR = gbrLonPCA.predict(xTest)



dists = EucDist(latPred, lonPred, latTest, lonTest)

meanED = np.mean(dists)

maxED = np.max(dists)

minED = np.min(dists)

print("--Metrics for SVR--")

print("meanED = " + str(meanED) + " m")

print("maxED = " + str(maxED) + " m")

print("minED = " + str(minED) + " m")



distsGBR = EucDist(latPredGBR, lonPredGBR, latTest, lonTest)

meanED = np.mean(distsGBR)

maxED = np.max(distsGBR)

minED = np.min(distsGBR)

print("--Metrics for GBR--")

print("meanED = " + str(meanED) + " m")

print("maxED = " + str(maxED) + " m")

print("minED = " + str(minED) + " m")


