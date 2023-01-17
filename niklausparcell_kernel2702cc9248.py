import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split, cross_val_predict

from sklearn.cross_decomposition import PLSRegression

from sklearn.metrics import mean_squared_error, r2_score 

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier, XGBRegressor, plot_importance

import math

import os

import sys 
# Import from .csv files

trainData = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv") 

testData = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv") 



# Create pandas dataFrames from .csv's 

trainData = pd.DataFrame(trainData) 

testData = pd.DataFrame(testData) 



# There are some empty cells in the .csv's. Fill those with 0's 

trainData.fillna("0", inplace = True) 

testData.fillna("0", inplace = True) 
class wordToNumber:

    """

    Index each word as a number, then return that list of words->numbers

    - "data" input is the column list from train or test data 

    """

    def returnWordToNumberList(self, data): 

        word_to_index = {} 

        index = 1 

        returnedList = [] 

        for word in data: 

            if word in word_to_index: 

                returnedList.append(word_to_index[word]) 

            else: 

                word_to_index[word] = index 

                returnedList.append(index) 

                index += 1

        return returnedList 
# Columns to be tokenized. Also including Date because dates show up multiple times

tokenizeColumns = [

    "Province/State", 

    "Country/Region", 

    "Date", 

]



# Algorithm 

for column in tokenizeColumns:

    trainData[column] = wordToNumber().returnWordToNumberList(data = trainData[column]) 

    testData[column] = wordToNumber().returnWordToNumberList(data = testData[column]) 
trainData = trainData.set_index("Id") 

testData = testData.set_index("ForecastId") 
# Define Y label columns

Y_columns = [

    "ConfirmedCases",

    "Fatalities",

]



# Split in to X and Y for train/validation sets

trainX, trainY = {}, {} 

for column in trainData.columns:

    if column in Y_columns:

        trainY[column] = trainData[column] 

    else:

        trainX[column] = trainData[column] 

        

# Make pandas data frames

trainX, trainY = pd.DataFrame(trainX), pd.DataFrame(trainY)
trainX, validationX, trainY, validationY = train_test_split(trainX, trainY, test_size = 0.2) 
# Need to use a dictionary for fitting

xgb_fit_graph = {

    "ConfirmedCases" : 0,

    "Fatalities" : 0,

}

for label in xgb_fit_graph:

    xgb = XGBRegressor(n_classifiers = 1000)

    xgb_fit_graph[label] = xgb.fit(trainX, trainY[label]) 



# Also use a dictionary for predictions

xgb_predictions = {}

for label in xgb_fit_graph:

    xgb_predictions[label + "Train"] = xgb_fit_graph[label].predict(trainX)

    xgb_predictions[label + "Validation"] = xgb_fit_graph[label].predict(validationX)

    
RF_fit_graph = {

    "ConfirmedCases" : 0,

    "Fatalities" : 0,

}

for label in RF_fit_graph:

    RF = RandomForestClassifier()

    RF.fit(trainX, trainY[label]) 

    RF_fit_graph[label] = RF 

    

# Dictionary for predictions

RF_predictions = {}

for label in RF_fit_graph:

    RF_predictions[label + "Train"] = RF_fit_graph[label].predict(trainX) 

    RF_predictions[label + "Validation"] = RF_fit_graph[label].predict(validationX) 
def geometricMean(values):

    """

    Found this equation from https://en.wikipedia.org/wiki/Geometric_mean

    """

    multipliedVal = 1 

    for element in values:

        multipliedVal *= element 

    

    N = len(values) 

    

    try:  # Sometimes error with values/doesn't like inputs

        multipliedVal = (multipliedVal) ** (1 / N) 

        if math.isnan(multipliedVal) is True: 

            multipliedVal = 0 

    except: 

        multipliedVal = 0 

        

    return multipliedVal
# Start with empty graph 

fusion_predictions = {}



# Define which regressors to be analyzed for their RMSEs 

fitGraphs = {

    "xgb" : xgb_fit_graph, 

    "RF" : RF_fit_graph,

}

regressorGraph = {

    "xgb" : xgb_predictions, 

    "RF" : RF_predictions,

}



# Types of sets 

setTypes = {

    "Train" : trainY, 

    "Validation" : validationY, 

}



# Fusion algorithm 

for label in xgb_fit_graph: 

    for setType in setTypes:

        temporaryArray = [] 

        for k, element in enumerate(regressorGraph["xgb"][label + setType]):

            values = [] 

            for regressor in regressorGraph:

                value = regressorGraph[regressor][label + setType][k] 

                values.append(value) 

            temporaryArray.append(geometricMean(values))

            

        fusion_predictions[label + setType] = temporaryArray



# Update regressorGraph

regressorGraph["Fusion"] = fusion_predictions

    
# Let's see how important variables are using gradient boost module

for label in xgb_fit_graph:

    plot_importance(xgb_fit_graph[label]) 

    plt.show()
for label in xgb_fit_graph:

    plt.figure(1)

    xlin = np.arange(min(trainY[label]) - 100, max(trainY[label]) + 100)

    plt.scatter(trainY[label], xgb_predictions[label+"Train"], label = "XGBR") 

    plt.scatter(trainY[label], RF_predictions[label+"Train"], label = "Random Forest")

    plt.scatter(trainY[label], fusion_predictions[label+"Train"], label = "Fusion")

    plt.plot(xlin, xlin) 

    plt.legend()

    plt.xlabel("Actual") 

    plt.ylabel("Prediction") 

    plt.title(label + " Train Predictions vs. Actual") 

    plt.show()

    

    plt.figure(2)

    xlin = np.arange(min(trainY[label]) - 100, max(trainY[label]) + 100)

    plt.scatter(validationY[label], xgb_predictions[label+"Validation"], label = "XGBR") 

    plt.scatter(validationY[label], RF_predictions[label+"Validation"], label = "RF")

    plt.scatter(validationY[label], fusion_predictions[label+"Validation"], label = "Fusion")

    plt.plot(xlin, xlin) 

    plt.legend()

    plt.xlabel("Actual") 

    plt.ylabel("Prediction") 

    plt.title(label + " Validation Predictions vs. Actual") 

    plt.show()
def RMSE(data1, data2):

    return mean_squared_error(data1, data2) ** (1/2)
# Start with empty graph

RMSEs = {} 



# Iterate through train and validation labels and update RMSE dictionary

for label in xgb_fit_graph: # Defined a few boxes above

    for regressor in regressorGraph: # also defined a few boxes above

        for setType in setTypes: # also defined a few boxes above

            RMSEs[label + regressor + setType] = RMSE(setTypes[setType][label], regressorGraph[regressor][label + setType])

    

# Print each RMSE

for label in RMSEs:

    print(label + " RMSE: %0.3f"%RMSEs[label])
submissionsCSV = {} 



# Test Predictions - fusion 

testPredictions = {} 



# Iterate through labels, and predict test data 

for label in xgb_fit_graph:

    tempGraph = {}

    for regressor in regressorGraph:

        if "Fusion" != regressor:

            tempGraph[regressor] = fitGraphs[regressor][label].predict(testData)



    testPredictions[label] = tempGraph



    predictions = []



    for k, element in enumerate(testPredictions[label]["xgb"]):

        values = []

        for regressor in regressorGraph:

            if "Fusion" != regressor:

                values.append(testPredictions[label][regressor][k]) 

        predictions.append(geometricMean(values))

    submissionsCSV[label] = predictions

    N = len(submissionsCSV[label])  # quick dirty solution to set forecastID, sorry

    

# Index 

submissionsCSV["ForecastId"] = np.arange(1, N+1) 



# Make data frame 

submissionsCSV = pd.DataFrame(submissionsCSV) 

submissionsCSV = submissionsCSV.set_index("ForecastId")



# Save data frame to submissions.csv 

submissionsCSV.to_csv("submission.csv")