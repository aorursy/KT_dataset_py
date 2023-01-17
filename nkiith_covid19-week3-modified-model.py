import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print('--------------------------------------')
print('Train data looks like ...')
trainData = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
print(trainData.head(5))

print('Test data looks like ...')
testData = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
print(testData.head(5))
# FOR KAGGLE COMPETITION : WEEK 3
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

    print('Are the Kaggle submission files same? ', np.shape(dfTrain), np.shape(dfTest), np.shape(result))
    return result


# Finding RMSE
def ErrorCalc(mdl, ref, tag):
    relError = np.abs(mdl - ref)/ np.abs(ref+1)
    MeanErrorV = np.mean(relError)
    print(tag + ': Mean Rel Error in %: ', MeanErrorV * 100)
    return MeanErrorV


# Since cumulative prediction >> This script is not used for Kaggle dataset
def AdjustingErrorsOutliers(tempPred, tempPrev) :
    tempPred = np.round(tempPred)
    for i in range(len(tempPred)):
        if tempPred[i] < tempPrev[i] : # Since cumulative prediction
            tempPred[i] = tempPrev[i]
    return tempPred




# Train model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def TrainMdl (trainIpData, trainOpData) :
    testSize = 0.1 # 90:10 ratio >> for final testing
    print('Training starts ...')

    randomState = None # For random train test split
    # randomState = 42 # For fixed train test split

    # Final validation
    X_train, X_test, y_train, y_test = train_test_split(trainIpData, trainOpData, test_size=testSize, random_state=randomState)



    # Set 1 - Infected feature dataset
    TrainIP1 = X_train[['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',
                    'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I']]
    TrainOP1 = X_train['gammaFunI']
    TestIP1 = X_test[['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',
                    'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I']]

    # Adaboost Model 1 for infection rate prediction
    treeDepth = 10 # Fixed
    mdl1 = DecisionTreeRegressor(max_depth=treeDepth) # This is fixed
    param_grid = {
        'n_estimators': [100, 250, 500],
        'learning_rate': [0.1, 0.01, 0.001]
                    }
    regrMdl1 = AdaBoostRegressor(base_estimator=mdl1)
    clf1 = RandomizedSearchCV(estimator = regrMdl1, param_distributions = param_grid,
                                         n_iter = 100, cv = 3, verbose=0, random_state=42, n_jobs = -1)
    clf1.fit(TrainIP1, TrainOP1)

    # Calculating Error
    y_predictedTrain = clf1.predict(TrainIP1) # Gamma
    y_predictedTrain = AdjustingErrorsOutliers(y_predictedTrain * TrainIP1['day5_I'].to_numpy(), TrainIP1['day5_I'].to_numpy()) # Returns after adjustment

    ErrorCalc(y_predictedTrain, X_train['dayPredictInf'].to_numpy(), 'Train Data-set model-1 (infection rate)') # y_predictedTrain converted to numbers

    y_predictedTest = clf1.predict(TestIP1) # Gamma
    y_predictedTest = AdjustingErrorsOutliers(y_predictedTest * TestIP1['day5_I'].to_numpy(), TestIP1['day5_I'].to_numpy()) # Returns after adjustment
    ErrorCalc(y_predictedTest, X_test['dayPredictInf'].to_numpy(), 'Test Data-set model-1 (infection rate)') # y_predictedTest converted to numbers



    # Set 2 - Fatality
    TrainIP2 = X_train[['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',
                    'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I',
                    'day1_F', 'day2_F', 'day3_F', 'day4_F', 'day5_F',
                    'diff1_F', 'diff2_F', 'diff3_F', 'diff4_F']]
    TrainOP2 = X_train['gammaFunF']
    TestIP2 = X_test[['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',
                    'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I',
                    'day1_F', 'day2_F', 'day3_F', 'day4_F', 'day5_F',
                    'diff1_F', 'diff2_F', 'diff3_F', 'diff4_F']]


    # Adaboost Model 2 for infection rate prediction
    treeDepth = 10  # Fixed
    mdl2 = DecisionTreeRegressor(max_depth=treeDepth)  # This is fixed
    param_grid = {
        'n_estimators': [100, 250, 500],
        'learning_rate': [0.1, 0.01, 0.001]
    }
    regrMdl2 = AdaBoostRegressor(base_estimator=mdl2)
    clf2 = RandomizedSearchCV(estimator=regrMdl2, param_distributions=param_grid,
                              n_iter=100, cv=3, verbose=0, random_state=42, n_jobs=-1)
    clf2.fit(TrainIP2, TrainOP2)

    # Calculating Error
    y_predictedTrain = clf2.predict(TrainIP2) * TrainIP2['day5_F'].to_numpy()  # Predicting the gamma function
    y_predictedTrain = AdjustingErrorsOutliers(y_predictedTrain, TrainIP2['day5_F'].to_numpy()) # Returns after adjustment
    ErrorCalc(y_predictedTrain, y_train.to_numpy(), 'Train Data-set model-2')  # y_predictedTrain converted to numbers

    y_predictedTest = clf2.predict(TestIP2) * TestIP2['day5_F'].to_numpy()  # Predicting the gamma function
    y_predictedTest = AdjustingErrorsOutliers(y_predictedTest, TestIP2['day5_F'].to_numpy()) # Returns after adjustment
    ErrorCalc(y_predictedTest, y_test.to_numpy(), 'Test Data-set model-2')  # y_predictedTest converted to numbers



    # Validation starts
    dfValidation_Inf = pd.read_csv("../input/week3-covid19-traindata/Validation_Infected.csv").reset_index(drop=True)
    dfValidation_Fat = pd.read_csv("../input/week3-covid19-traindata/Validation_Fatality.csv").reset_index(drop=True)

    selRow = 0 # Row count
    startIdx = 12 # To eliminate previous columns

    # Size of the array
    [rVal, cVal] = np.shape(dfValidation_Inf) # Both the validation data-sets are same
    lengthZ = cVal-1
    lengthZ = 12 * lengthZ
    arrP = np.zeros((2 * lengthZ,))
    arrA = np.zeros((2 * lengthZ,))
    count = 0
    error = 0
    errorLen = 0

    # Kickstart
    Threshold_I = 0
    Threshold_F = 0

    print('Validating ...')

    while (selRow < rVal): # Scans the rows
        # print(count)
        iDetect = startIdx # Starts column scan
        iArray = 0
        #while (iDetect < cVal-1): # Scans until the last column
        while (iDetect < 17):  # Scans until the last column
            if iDetect == startIdx :
                day5_I = dfValidation_Inf.iloc[selRow, iDetect]
                day4_I = dfValidation_Inf.iloc[selRow, iDetect - 1]
                day3_I = dfValidation_Inf.iloc[selRow, iDetect - 2]
                day2_I = dfValidation_Inf.iloc[selRow, iDetect - 3]
                day1_I = dfValidation_Inf.iloc[selRow, iDetect - 4]

                # Kickstart
                if day5_I < Threshold_I:
                    day5_I = Threshold_I

                day5_F = dfValidation_Fat.iloc[selRow, iDetect]
                day4_F = dfValidation_Fat.iloc[selRow, iDetect - 1]
                day3_F = dfValidation_Fat.iloc[selRow, iDetect - 2]
                day2_F = dfValidation_Fat.iloc[selRow, iDetect - 3]
                day1_F = dfValidation_Fat.iloc[selRow, iDetect - 4]

                # Kickstart
                if day5_F < Threshold_F:
                    day5_F = Threshold_F

            else:
                day1_I = day2_I
                day2_I = day3_I
                day3_I = day4_I
                day4_I = day5_I
                day5_I = predictedInfected

                day1_F = day2_F
                day2_F = day3_F
                day3_F = day4_F
                day4_F = day5_F
                day5_F = predictedFatality

            # Run time calculation of other features
            diff1_I = day5_I - day4_I
            diff2_I = day4_I - day3_I
            diff3_I = day3_I - day2_I
            diff4_I = day2_I - day1_I

            diff1_F = day5_F - day4_F
            diff2_F = day4_F - day3_F
            diff3_F = day3_F - day2_F
            diff4_F = day2_F - day1_F


            data1 = {'day1_I': [day1_I], 'day2_I': [day2_I], 'day3_I': [day3_I], 'day4_I': [day4_I], 'day5_I': [day5_I],
                     'diff1_I': [diff1_I], 'diff2_I': [diff2_I], 'diff3_I': [diff3_I], 'diff4_I': [diff4_I]}
            dfPredict1 = pd.DataFrame(data1)
            predictedInfected = clf1.predict(dfPredict1[['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',
                     'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I']]) * day5_I
            if predictedInfected < day5_I:
                predictedInfected = day5_I
            actVal = dfValidation_Inf.iloc[selRow, iDetect + 1]
            arrP[iArray] = predictedInfected
            arrA[iArray] = actVal
            iArray = iArray + 1


            data2 = {'day1_I': [day1_I], 'day2_I': [day2_I], 'day3_I': [day3_I], 'day4_I': [day4_I], 'day5_I': [day5_I],
                    'diff1_I': [diff1_I], 'diff2_I': [diff2_I], 'diff3_I': [diff3_I], 'diff4_I': [diff4_I],
                    'day1_F': [day1_F], 'day2_F': [day2_F], 'day3_F': [day3_F], 'day4_F': [day4_F], 'day5_F': [day5_F],
                    'diff1_F': [diff1_F], 'diff2_F': [diff2_F], 'diff3_F': [diff3_F], 'diff4_F': [diff4_F]}
            dfPredict2 = pd.DataFrame(data2)
            predictedFatality = clf2.predict(dfPredict2[['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',
                    'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I',
                    'day1_F', 'day2_F', 'day3_F', 'day4_F', 'day5_F',
                    'diff1_F', 'diff2_F', 'diff3_F', 'diff4_F']]) * day5_F
            if predictedFatality < day5_F:
                predictedFatality = day5_F
            actVal = dfValidation_Fat.iloc[selRow, iDetect + 1]
            arrP[iArray] = predictedFatality
            arrA[iArray] = actVal

            iDetect = iDetect + 1
            iArray = iArray + 1

        # For each row
        error = error + sum(np.square(np.log(arrP[0:iArray-1] + 1) - np.log(arrA[0:iArray-1] + 1)))
        errorLen = errorLen + iArray #

        selRow = selRow + 1  # Move to the next row
        count = count + 1


    # Final error
    error = float(error) /  errorLen
    print('Validation error: ', error)


    print('Making Kaggle Submission file ...')

    # Read Kaggle dataset
    dfTrain = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
    dfTest = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
    df = kaggle(dfTrain, dfTest) # To pad data


    # Kaggle submission
    print('Prediction starts for Kaggle submission ...')
    [rr, cc] = np.shape(df)
    for iP in range(rr):
        if df.loc[iP, 'ConfirmedCases'] == -1 : # iP-th position need to be predicted
            # Create a dataframe
            day5_I = df.loc[iP-1, 'ConfirmedCases']
            day4_I = df.loc[iP-2, 'ConfirmedCases']
            day3_I = df.loc[iP-3, 'ConfirmedCases']
            day2_I = df.loc[iP-4, 'ConfirmedCases']
            day1_I = df.loc[iP-5, 'ConfirmedCases']

            day5_F = df.loc[iP - 1, 'Fatalities']
            day4_F = df.loc[iP - 2, 'Fatalities']
            day3_F = df.loc[iP - 3, 'Fatalities']
            day2_F = df.loc[iP - 4, 'Fatalities']
            day1_F = df.loc[iP - 5, 'Fatalities']

            # Run time calculation of other features
            diff1_I = day5_I - day4_I
            diff2_I = day4_I - day3_I
            diff3_I = day3_I - day2_I
            diff4_I = day2_I - day1_I

            diff1_F = day5_F - day4_F
            diff2_F = day4_F - day3_F
            diff3_F = day3_F - day2_F
            diff4_F = day2_F - day1_F


            data1 = {'day1_I': [day1_I], 'day2_I': [day2_I], 'day3_I': [day3_I], 'day4_I': [day4_I], 'day5_I': [day5_I],
                     'diff1_I': [diff1_I], 'diff2_I': [diff2_I], 'diff3_I': [diff3_I], 'diff4_I': [diff4_I]}
            dfPredict1 = pd.DataFrame(data1)
            predictedInfected = clf1.predict(dfPredict1[['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',
                     'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I']]) * day5_I
            if predictedInfected < day5_I:
                predictedInfected = day5_I

            df.loc[iP, 'ConfirmedCases'] = np.round(predictedInfected)  # Update the current location



            data2 = {'day1_I': [day1_I], 'day2_I': [day2_I], 'day3_I': [day3_I], 'day4_I': [day4_I], 'day5_I': [day5_I],
                    'diff1_I': [diff1_I], 'diff2_I': [diff2_I], 'diff3_I': [diff3_I], 'diff4_I': [diff4_I],
                    'day1_F': [day1_F], 'day2_F': [day2_F], 'day3_F': [day3_F], 'day4_F': [day4_F], 'day5_F': [day5_F],
                    'diff1_F': [diff1_F], 'diff2_F': [diff2_F], 'diff3_F': [diff3_F], 'diff4_F': [diff4_F]}
            dfPredict2 = pd.DataFrame(data2)
            predictedFatality = clf2.predict(dfPredict2[['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',
                    'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I',
                    'day1_F', 'day2_F', 'day3_F', 'day4_F', 'day5_F',
                    'diff1_F', 'diff2_F', 'diff3_F', 'diff4_F']]) * day5_F
            if predictedFatality < day5_F:
                predictedFatality = day5_F

            df.loc[iP, 'Fatalities'] = np.round(predictedFatality) # Update the current location
    return df




# Main code starts >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
df = pd.read_csv("../input/week3-covid19-traindata/TrainTest.csv") # Processed dta from JHU

trainIpData = df[['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',
                    'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I',
                    'day1_F', 'day2_F', 'day3_F', 'day4_F', 'day5_F',
                    'diff1_F', 'diff2_F', 'diff3_F', 'diff4_F', 'gammaFunF', 'gammaFunI', 'dayPredictInf']]
trainOpData = df['dayPredictFat'] # 'dayPredictInf' is fed to the input set
predictions_dF = TrainMdl (trainIpData, trainOpData) # Kaggle data will be read inside the function

predictions_dF[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv('submission.csv', index=False)
print(predictions_dF[['ForecastId', 'ConfirmedCases', 'Fatalities']].head(5))
print(predictions_dF[['ForecastId', 'ConfirmedCases', 'Fatalities']].tail(5))

print('Done!')