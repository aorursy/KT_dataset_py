import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print('--------------------------------------')

print('Train data looks like ...')

trainData = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

print(trainData.head(5))



print('Test data looks like ...')

testData = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")

print(testData.head(5))
# KAGGLE competition >> Prediction of confirmed cases

# This script trains the model on the latest dataset and predicts the next value

# Author: Neilay Khasnabish





#  Import libraries

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor

import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV

import tqdm as tqdm



# Making Kaggle dataset ready

def kaggle(dfTrain, dfTest):

    pd.set_option('display.max_columns', None)



    dfTest['DateNew'] = pd.to_datetime(dfTest['Date'])

    dfTest = dfTest.drop(['Date'], axis=1)

    dfTest = dfTest.rename(columns={"DateNew": "Date"})

    dfTest['Year'] = dfTest['Date'].dt.year

    dfTest['Month'] = dfTest['Date'].dt.month

    dfTest['Day'] = dfTest['Date'].dt.day

    dfTest = dfTest.drop(['Date'], axis=1)

    dfTest = dfTest.fillna('DummyProvince')

    #dfTest.to_csv('G:/Kaggle_COVID/covid19-global-forecasting-week-2/dummyTest.csv')



    dfTrain['DateNew'] = pd.to_datetime(dfTrain['Date'])

    dfTrain = dfTrain.drop(['Date'], axis=1)

    dfTrain = dfTrain.rename(columns={"DateNew": "Date"})

    dfTrain['Year'] = dfTrain['Date'].dt.year

    dfTrain['Month'] = dfTrain['Date'].dt.month

    dfTrain['Day'] = dfTrain['Date'].dt.day

    dfTrain = dfTrain.drop(['Date'], axis=1)

    dfTrain = dfTrain.fillna('DummyProvince')

    #dfTrain.to_csv('G:/Kaggle_COVID/covid19-global-forecasting-week-2/dummyTrain.csv')



    result = pd.merge(dfTest, dfTrain, how='left', on=['Country_Region', 'Province_State', 'Year', 'Month', 'Day'])

    result = result.fillna(-1)



    # Clutter removal

    [rr, cc] = np.shape(result)

    for iQuit in range(rr):

        if result.loc[iQuit, 'Day'] == 4 :

            result.loc[iQuit, 'ConfirmedCases'] = -1

            result.loc[iQuit, 'Fatalities'] = -1



    #result.to_csv('G:/Kaggle_COVID/covid19-global-forecasting-week-2/temp.csv')

    return result





# Finding RMSE

def ErrorCalc(mdl, ref, tag):

    relError = np.abs(mdl - ref)/ np.abs(ref+1)

    MeanErrorV = np.mean(relError)

    print(tag + ': Mean Rel Error in %: ', MeanErrorV * 100)

    return MeanErrorV





# Since cumulative prediction >> This script is not used for Kaggle dataset

def AdjustingErrorsOutliers(tempPred, df) :

    tempPred = np.round(tempPred)

    tempPrev = df['day5'].to_numpy() # Next cumulative prediction must be more than or equal to previous

    for i in range(len(tempPred)):

        if tempPred[i] < tempPrev[i] : # Since cumulative prediction

            tempPred[i] = tempPrev[i]

    return tempPred





# Train model

def TrainMdl (trainIpData, trainOpData) :



    testSize = 0.1 # 90:10 ratio >> for final testing



    print('Training starts ...')



    randomState=None

    # randomState = 42 # For train test split



    # Final validation

    X_train, X_test, y_train, y_test = train_test_split(trainIpData, trainOpData, test_size=testSize, random_state=randomState)



    # Another set of input

    TrainIP = X_train[['day1', 'day2', 'day3', 'day4', 'day5', 'diff1', 'diff2', 'diff3', 'diff4']]

    TrainOP = X_train['gammaFun']

    TestIP = X_test[['day1', 'day2', 'day3', 'day4', 'day5', 'diff1', 'diff2', 'diff3', 'diff4']]

    TestOP = X_test['gammaFun']





    # Adaboost Regressor >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    treeDepth = 10 # Fixed

    mdl = DecisionTreeRegressor(max_depth=treeDepth) # This is fixed

    param_grid = {

        'n_estimators': [100, 250, 500],

        'learning_rate': [0.1, 0.01, 0.001]

                    }

    regrMdl = AdaBoostRegressor(base_estimator=mdl)

    clf = RandomizedSearchCV(estimator = regrMdl, param_distributions = param_grid,

                                         n_iter = 100, cv = 3, verbose=0, random_state=42, n_jobs = -1)

    clf.fit(TrainIP, TrainOP)





    # Calculating Error

    y_predictedTrain = clf.predict(TrainIP) # Predicting the gamma function

    y_predictedTrain = AdjustingErrorsOutliers(y_predictedTrain * TrainIP['day5'].to_numpy(), TrainIP)

    ErrorCalc(y_predictedTrain, y_train.to_numpy(), 'Train Data-set') # y_predictedTrain converted to numbers



    y_predictedTest = clf.predict(TestIP) # Predicting the gamma function

    y_predictedTest = AdjustingErrorsOutliers(y_predictedTest * TestIP['day5'].to_numpy(), TestIP)

    ErrorCalc(y_predictedTest, y_test.to_numpy(), 'Validation Data-set ') # y_predictedTest converted to numbers



    print('-----------------------------------------------------------')



    # Read Kaggle dataset

    dfTrain = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

    dfTest = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")

    df = kaggle(dfTrain, dfTest)



    print('Prediction starts ...')

    [rr, cc] = np.shape(df)

    for iP in range(rr):

        if df.loc[iP, 'ConfirmedCases'] == -1 : # iP-th position need to be predicted

            # Create a dataframe

            day5 = df.loc[iP-1, 'ConfirmedCases']

            day4 = df.loc[iP-2, 'ConfirmedCases']

            day3 = df.loc[iP-3, 'ConfirmedCases']

            day2 = df.loc[iP-4, 'ConfirmedCases']

            day1 = df.loc[iP-5, 'ConfirmedCases']

            diff1 = day5 - day4

            diff2 = day4 - day3

            diff3 = day3 - day2

            diff4 = day2 - day1

            data = {'day1': [day1], 'day2': [day2], 'day3': [day3], 'day4': [day4], 'day5': [day5],

                    'diff1': [diff1], 'diff2': [diff2], 'diff3': [diff3], 'diff4': [diff4]}

            dfPredict = pd.DataFrame(data)

            finalPrediction = clf.predict(dfPredict[['day1', 'day2', 'day3', 'day4', 'day5', 'diff1', 'diff2', 'diff3', 'diff4']]) * day5

            if finalPrediction < day5 :

                finalPrediction = day5

            df.loc[iP, 'ConfirmedCases'] = np.round(finalPrediction) # Update the current location



    return df





# Main code starts

df = pd.read_csv("../input/processedtimedata/TrainTest.csv") # Processed dta from JHU

trainIpData = df[['day1', 'day2', 'day3', 'day4', 'day5', 'gammaFun', 'diff1', 'diff2', 'diff3', 'diff4']]

trainOpData = df['dayPredict'] # Predicted confirmed case

predictions_dF = TrainMdl (trainIpData, trainOpData) # Kaggle data will be read inside

print('Completed ...')



#predictions_dF[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv('G:/Kaggle_COVID/covid19-global-forecasting-week-2/submission_ConfirmedCases.csv', index = False)
# KAGGLE competition >> Fatality rate

# This script trains the model on the latest dataset and predicts the next value

# Author: Neilay Khasnabish





#  Import libraries

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor

import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV



# Making Kaggle dataset ready

def kaggle(dfTrain, dfTest):

    pd.set_option('display.max_columns', None)



    dfTest['DateNew'] = pd.to_datetime(dfTest['Date'])

    dfTest = dfTest.drop(['Date'], axis=1)

    dfTest = dfTest.rename(columns={"DateNew": "Date"})

    dfTest['Year'] = dfTest['Date'].dt.year

    dfTest['Month'] = dfTest['Date'].dt.month

    dfTest['Day'] = dfTest['Date'].dt.day

    dfTest = dfTest.drop(['Date'], axis=1)

    dfTest = dfTest.fillna('DummyProvince')

    #dfTest.to_csv('G:/Kaggle_COVID/covid19-global-forecasting-week-2/dummyTest.csv')



    dfTrain['DateNew'] = pd.to_datetime(dfTrain['Date'])

    dfTrain = dfTrain.drop(['Date'], axis=1)

    dfTrain = dfTrain.rename(columns={"DateNew": "Date"})

    dfTrain['Year'] = dfTrain['Date'].dt.year

    dfTrain['Month'] = dfTrain['Date'].dt.month

    dfTrain['Day'] = dfTrain['Date'].dt.day

    dfTrain = dfTrain.drop(['Date'], axis=1)

    dfTrain = dfTrain.fillna('DummyProvince')

    #dfTrain.to_csv('G:/Kaggle_COVID/covid19-global-forecasting-week-2/dummyTrain.csv')



    result = pd.merge(dfTest, dfTrain, how='left', on=['Country_Region', 'Province_State', 'Year', 'Month', 'Day'])

    result = result.fillna(-1)



    # Clutter removal

    [rr, cc] = np.shape(result)

    for iQuit in range(rr):

        if result.loc[iQuit, 'Day'] == 4 :

            result.loc[iQuit, 'ConfirmedCases'] = -1

            result.loc[iQuit, 'Fatalities'] = -1



    #result.to_csv('G:/Kaggle_COVID/covid19-global-forecasting-week-2/temp.csv')

    return result





# Finding RMSE

def ErrorCalc(mdl, ref, tag):

    relError = np.abs(mdl - ref)/ np.abs(ref+1)

    MeanErrorV = np.mean(relError)

    print(tag + ': Mean Rel Error in %: ', MeanErrorV * 100)

    return MeanErrorV





# Since cumulative prediction >> This script is not used for Kaggle dataset

def AdjustingErrorsOutliers(tempPred, df) :

    tempPred = np.round(tempPred)

    tempPrev = df['day5'].to_numpy() # Next cumulative prediction must be more than or equal to previous

    for i in range(len(tempPred)):

        if tempPred[i] < tempPrev[i] : # Since cumulative prediction

            tempPred[i] = tempPrev[i]

    return tempPred





# Train model

def TrainMdl (trainIpData, trainOpData) :







    testSize = 0.1 # 90:10 ratio >> for final testing



    print('Training starts ...')



    randomState=None

    # randomState = 42 # For train test split



    # Final validation

    X_train, X_test, y_train, y_test = train_test_split(trainIpData, trainOpData, test_size=testSize, random_state=randomState)



    # Another set of input

    TrainIP = X_train[['day1', 'day2', 'day3', 'day4', 'day5', 'diff1', 'diff2', 'diff3', 'diff4']]

    TrainOP = X_train['gammaFun']

    TestIP = X_test[['day1', 'day2', 'day3', 'day4', 'day5', 'diff1', 'diff2', 'diff3', 'diff4']]

    TestOP = X_test['gammaFun']





    # Adaboost Regressor >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    treeDepth = 10 # Fixed

    mdl = DecisionTreeRegressor(max_depth=treeDepth) # This is fixed

    param_grid = {

        'n_estimators': [100, 250, 500],

        'learning_rate': [0.1, 0.01, 0.001]

                    }

    regrMdl = AdaBoostRegressor(base_estimator=mdl)

    clf = RandomizedSearchCV(estimator = regrMdl, param_distributions = param_grid,

                                         n_iter = 100, cv = 3, verbose=0, random_state=42, n_jobs = -1)

    clf.fit(TrainIP, TrainOP)





    # Calculating Error

    y_predictedTrain = clf.predict(TrainIP) # Predicting the gamma function

    y_predictedTrain = AdjustingErrorsOutliers(y_predictedTrain * TrainIP['day5'].to_numpy(), TrainIP)

    ErrorCalc(y_predictedTrain, y_train.to_numpy(), 'Train Data-set') # y_predictedTrain converted to numbers



    y_predictedTest = clf.predict(TestIP) # Predicting the gamma function

    y_predictedTest = AdjustingErrorsOutliers(y_predictedTest * TestIP['day5'].to_numpy(), TestIP)

    ErrorCalc(y_predictedTest, y_test.to_numpy(), 'Validation Data-set ') # y_predictedTest converted to numbers



    print('-----------------------------------------------------------')





    # Read Kaggle dataset

    dfTrain = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

    dfTest = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")

    df = kaggle(dfTrain, dfTest)







    [rr, cc] = np.shape(df)

    for iP in range(rr):

        if df.loc[iP, 'Fatalities'] == -1 : # iP-th position need to be predicted

            # Create a dataframe

            day5 = df.loc[iP-1, 'Fatalities']

            day4 = df.loc[iP-2, 'Fatalities']

            day3 = df.loc[iP-3, 'Fatalities']

            day2 = df.loc[iP-4, 'Fatalities']

            day1 = df.loc[iP-5, 'Fatalities']

            diff1 = day5 - day4

            diff2 = day4 - day3

            diff3 = day3 - day2

            diff4 = day2 - day1

            data = {'day1': [day1], 'day2': [day2], 'day3': [day3], 'day4': [day4], 'day5': [day5],

                    'diff1': [diff1], 'diff2': [diff2], 'diff3': [diff3], 'diff4': [diff4]}

            dfPredict = pd.DataFrame(data)

            finalPrediction = clf.predict(dfPredict[['day1', 'day2', 'day3', 'day4', 'day5', 'diff1', 'diff2', 'diff3', 'diff4']]) * day5

            if finalPrediction < day5 :

                finalPrediction = day5

            df.loc[iP, 'Fatalities'] = np.round(finalPrediction) # Update the current location





    return df





# Main code starts

df =  pd.read_csv("../input/processedtimedata/TrainTest_Fatality.csv") # Processed dta from JHU

trainIpData = df[['day1', 'day2', 'day3', 'day4', 'day5', 'gammaFun', 'diff1', 'diff2', 'diff3', 'diff4']]

trainOpData = df['dayPredict'] # Predicted fatality

fatality_dF = TrainMdl (trainIpData, trainOpData) # Kaggle data will be read inside

print('Completed ...')

#predictions_dF[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv('G:/Kaggle_COVID/covid19-global-forecasting-week-2/submission_Fatality.csv', index = False)
# Creating the submission

predictions_dF['Fatalities'] = fatality_dF['Fatalities']

predictions_dF[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv('submission.csv', index=False)

print(predictions_dF[['ForecastId', 'ConfirmedCases', 'Fatalities']].head(10))

print(predictions_dF[['ForecastId', 'ConfirmedCases', 'Fatalities']].tail(10))