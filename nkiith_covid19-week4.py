import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print('--------------------------------------')

print('Train data looks like ...')

trainData = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

print(trainData.head(5))



print('Test data looks like ...')

testData = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

print(testData.head(5))
# FOR KAGGLE COMPETITION : WEEK 4

# This script can extract data from John Hopkins Dataset and make it ready

# https://github.com/CSSEGISandData/COVID-19

# Author: Neilay Khasnabish







# Lib

import pandas as pd

import numpy as np

from tqdm import tqdm





# Create dataset for both the cases (02 cases)

def createDataset2(resultInf, resultFat):

    print('Are they of same shape? : ', np.shape(resultInf), np.shape(resultFat))

    [rf, cf] = np.shape(resultInf)

    df=[]

    count = 0

    for i in tqdm(range(rf)): # It scans through the entire row

        #count = count + 1

        #print(count)

        iCol = 5 # Start index after the country

        while iCol < cf-1 :



            day5_I = resultInf.iloc[i, iCol] # 5-th

            day4_I = resultInf.iloc[i, iCol-1] # 4-th

            day3_I = resultInf.iloc[i, iCol-2] # 3-th

            day2_I = resultInf.iloc[i, iCol-3]# 2-nd

            day1_I = resultInf.iloc[i, iCol-4]# 1-st

            # 0-th >> Country name (older version)

            diff1_I = day5_I - day4_I

            diff2_I = day4_I - day3_I

            diff3_I = day3_I - day2_I

            diff4_I  = day2_I - day1_I



            day5_F = resultFat.iloc[i, iCol] # 5-th

            day4_F = resultFat.iloc[i, iCol-1] # 4-th

            day3_F = resultFat.iloc[i, iCol-2] # 3-th

            day2_F = resultFat.iloc[i, iCol-3]# 2-nd

            day1_F = resultFat.iloc[i, iCol-4]# 1-st

            # 0-th >> Country name (older version)

            diff1_F = day5_F - day4_F

            diff2_F = day4_F - day3_F

            diff3_F = day3_F - day2_F

            diff4_F  = day2_F - day1_F



            # Predict over a horizon

            horizon = int(iCol)

            time_horizon = 0

            while horizon < cf - 1:

                time_horizon =  time_horizon + 1

                horizon = horizon + 1



                dayPredictInf = resultInf.iloc[i, horizon]

                dayPredictFat = resultFat.iloc[i, horizon]



                data = {'day1_I': [day1_I], 'day2_I': [day2_I], 'day3_I': [day3_I], 'day4_I': [day4_I], 'day5_I': [day5_I],

                    'diff1_I': [diff1_I], 'diff2_I': [diff2_I], 'diff3_I': [diff3_I], 'diff4_I': [diff4_I],

                    'day1_F': [day1_F], 'day2_F': [day2_F], 'day3_F': [day3_F], 'day4_F': [day4_F], 'day5_F': [day5_F],

                    'diff1_F': [diff1_F], 'diff2_F': [diff2_F], 'diff3_F': [diff3_F], 'diff4_F': [diff4_F],

                    'dayPredictFat': [dayPredictFat], 'dayPredictInf': [dayPredictInf],  'time_horizon': [time_horizon]}



                df2 = pd.DataFrame(data)

                df.append(df2)



            iCol = iCol + 1



    df = pd.concat(df).reset_index(drop=True)

    df = df.fillna(0) # Actually not needed

    df.to_csv('TrainTest.csv', index=False)

    #print(df.head(5))













# Main script starts here >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Settings

displayAll = 0

if displayAll == 1 :

    pd.set_option('display.max_columns', None)



# Set path

readDataPath1 = "../input/jhudataset/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

readDataPath2 = "../input/jhudataset/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

print('Are the of same shape? ', np.shape(pd.read_csv(readDataPath1)), np.shape(pd.read_csv(readDataPath2)))



# Infection case : Country-wise

worldCorona = pd.read_csv(readDataPath1)

worldCorona = worldCorona.fillna(0)

worldCorona['Country'] = worldCorona['Country/Region']

worldCorona = worldCorona.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)

worldCorona = worldCorona.groupby(['Country']).sum()

worldCorona.to_csv('HellosInfected.csv')

worldCorona = pd.read_csv('HellosInfected.csv').reset_index(drop=True) # Will make validation data out of it

result = worldCorona.drop([0, 9, 154]).reset_index(drop=True) # Dropped for validation purpose

result = result.drop(['Country'], axis=1)

print('Size of country-wise data (infected): ', np.shape(result))



# Infection case : province-wise

worldCoronaProvince = pd.read_csv(readDataPath1)

worldCoronaProvince = worldCoronaProvince.dropna().reset_index(drop=True) # Drop rows having NaN >> Means no province

worldCoronaProvince = worldCoronaProvince.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)

print('Size of province-wise data (infected): ', np.shape(worldCoronaProvince))



# Appending : Country and province

resultConfirmed = result.append(worldCoronaProvince)



# Creating dataset for validation  with country and province data

dfVal = resultConfirmed.copy()

dfVal = dfVal.drop(dfVal.iloc[:, 1:48], axis=1)

dfVal.to_csv('Validation_Infected.csv', index=False)







# Fatality case : Country-wise

worldCorona = pd.read_csv(readDataPath2)

worldCorona = worldCorona.fillna(0)

worldCorona['Country'] = worldCorona['Country/Region']

worldCorona = worldCorona.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)

worldCorona = worldCorona.groupby(['Country']).sum()

worldCorona.to_csv('HellosFatality.csv')

worldCorona = pd.read_csv('HellosFatality.csv').reset_index(drop=True) # Will make validation data out of it

result = worldCorona.drop([0, 9, 154]).reset_index(drop=True) # Dropped for validation purpose

result = result.drop(['Country'], axis=1)

print('Size of country-wise data (fatality): ', np.shape(result))



# Fatality case : Province-wise

worldCoronaProvince = pd.read_csv(readDataPath2)

worldCoronaProvince = worldCoronaProvince.dropna().reset_index(drop=True) # Drop rows having NaN

worldCoronaProvince = worldCoronaProvince.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)

print('Size of province-wise data (fatality): ', np.shape(worldCoronaProvince))



# Appending : Country and province

resultFatality = result.append(worldCoronaProvince)



# Creating dataset for validation  with country and province data

dfVal = resultFatality.copy()

dfVal = dfVal.drop(dfVal.iloc[:, 1:48], axis=1)

dfVal.to_csv('Validation_Fatality.csv', index=False)



# Creating dataset for training

createDataset2(resultConfirmed, resultFatality)
# Model selection



import pandas as pd

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split



df = pd.read_csv('TrainTest.csv') # Processed data from JHU



print('Size of training data: ', np.shape(df))

df = df.groupby(['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',

                    'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I',

                    'day1_F', 'day2_F', 'day3_F', 'day4_F', 'day5_F',

                    'diff1_F', 'diff2_F', 'diff3_F', 'diff4_F', 'time_horizon']).mean()

df.to_csv('temp2.csv')

df = pd.read_csv('temp2.csv').reset_index(drop=True) # Will make validation data out of it



print('Size of training data after grouping: ', np.shape(df))





# Estimating trajectories

dfExtra = df[df['day1_I'] > 80000].reset_index(drop=True).copy() # Threshold as per JHU dataset

dfExtra = dfExtra.groupby(['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',

                    'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I',

                    'day1_F', 'day2_F', 'day3_F', 'day4_F', 'day5_F',

                    'diff1_F', 'diff2_F', 'diff3_F', 'diff4_F']).max()

dfExtra.to_csv('temp4.csv')

dfExtra = pd.read_csv('temp4.csv').reset_index(drop=True)

dfExtra = dfExtra[dfExtra['time_horizon'] == 1].reset_index(drop=True)

print('Extra data shape', np.shape(dfExtra))

[rExtra, cExtra] = np.shape(dfExtra)

dfNew=[]

for iNew in range(rExtra):

    time_horizon = 1

    

    day5_I = dfExtra.loc[iNew, 'day5_I'] # 5-th

    day4_I = dfExtra.loc[iNew, 'day4_I'] # 4-th

    day3_I = dfExtra.loc[iNew, 'day3_I'] # 3-th

    day2_I = dfExtra.loc[iNew, 'day2_I']# 2-nd

    day1_I = dfExtra.loc[iNew, 'day1_I']# 1-st

    diff1_I = dfExtra.loc[iNew, 'diff1_I']

    diff2_I = dfExtra.loc[iNew, 'diff2_I']

    diff3_I = dfExtra.loc[iNew, 'diff3_I']

    diff4_I  = dfExtra.loc[iNew, 'diff4_I']



    day5_F = dfExtra.loc[iNew, 'day5_F'] # 5-th

    day4_F = dfExtra.loc[iNew, 'day4_F'] # 4-th

    day3_F = dfExtra.loc[iNew, 'day3_F'] # 3-th

    day2_F = dfExtra.loc[iNew, 'day2_F']# 2-nd

    day1_F = dfExtra.loc[iNew, 'day1_F']# 1-st

    diff1_F = dfExtra.loc[iNew, 'diff1_F']

    diff2_F = dfExtra.loc[iNew, 'diff1_F']

    diff3_F = dfExtra.loc[iNew, 'diff1_F']

    diff4_F  = dfExtra.loc[iNew, 'diff1_F']  

        

    stepI = diff4_I # Constant

    stepF = diff4_F # Constant

        

    while time_horizon < 35 : # Define horizon

        time_horizon = time_horizon + 1

    

        stepC = int(time_horizon - 1)

        stepI = float(stepI * 0.98) # High in momentum

        stepF = float(stepF * 0.96) # High in momentum

        deltaI = stepC * stepI 

        deltaF = stepC * stepF

    

        dayPredictFat  = dfExtra.loc[iNew, 'dayPredictFat'] + deltaF 

        dayPredictInf  = dfExtra.loc[iNew, 'dayPredictInf'] + deltaI

        

        # Make dataframe

        data = {'day1_I': [day1_I], 'day2_I': [day2_I], 'day3_I': [day3_I], 'day4_I': [day4_I], 'day5_I': [day5_I],

                    'diff1_I': [diff1_I], 'diff2_I': [diff2_I], 'diff3_I': [diff3_I], 'diff4_I': [diff4_I],

                    'day1_F': [day1_F], 'day2_F': [day2_F], 'day3_F': [day3_F], 'day4_F': [day4_F], 'day5_F': [day5_F],

                    'diff1_F': [diff1_F], 'diff2_F': [diff2_F], 'diff3_F': [diff3_F], 'diff4_F': [diff4_F],

                    'dayPredictFat': [dayPredictFat], 'dayPredictInf': [dayPredictInf],  'time_horizon': [time_horizon]}



        dfNewAdd = pd.DataFrame(data)

        dfNew.append(dfNewAdd)

        

dfNew = pd.concat(dfNew).reset_index(drop=True)

print('Size of extra data: ', np.shape(dfNew))

df = df.append(dfNew).reset_index(drop=True)



print('Size of training data after extra operation: ', np.shape(df))

tuneTrue = 0 # 1 for tuning the model; 0 for real prediction



if tuneTrue == 1:  # For tuning purpose

    testSize = 0.8 # Change depending on dataset

    randomState = None

    X_train1, X_test2= train_test_split(df, test_size=testSize, random_state=randomState)

    X_train1.to_csv('temp3.csv')

    df = pd.read_csv('temp3.csv').reset_index(drop=True)

    df = df.drop(['Unnamed: 0'], axis=1)



print('Size of training data after sorting: ', np.shape(df))

print('Is NaN? : ', df.isnull().values.any())
# FOR KAGGLE COMPETITION : WEEK 4

# This script trains the model on the latest dataset and predicts over the horizon

# Author: Neilay Khasnabish





#  Import libraries

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor

import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV

from tqdm import tqdm

import xgboost as xgb



# Making Kaggle dataset ready

def kaggle(dfTrain, dfTest):

    #print('Shape of Train', np.shape(dfTrain))

    #print('Shape of Test', np.shape(dfTest))

    pd.set_option('display.max_columns', None)



    dfTest['DateNew'] = pd.to_datetime(dfTest['Date'])

    dfTest = dfTest.drop(['Date'], axis=1)

    dfTest = dfTest.rename(columns={"DateNew": "Date"})

    dfTest['Year'] = dfTest['Date'].dt.year

    dfTest['Month'] = dfTest['Date'].dt.month

    dfTest['Day'] = dfTest['Date'].dt.day

    dfTest = dfTest.drop(['Date'], axis=1)

    dfTest = dfTest.fillna('DummyProvince')



    dfTrain['DateNew'] = pd.to_datetime(dfTrain['Date'])

    dfTrain = dfTrain.drop(['Date'], axis=1)

    dfTrain = dfTrain.rename(columns={"DateNew": "Date"})

    dfTrain['Year'] = dfTrain['Date'].dt.year

    dfTrain['Month'] = dfTrain['Date'].dt.month

    dfTrain['Day'] = dfTrain['Date'].dt.day

    dfTrain = dfTrain.drop(['Date'], axis=1)

    dfTrain = dfTrain.fillna('DummyProvince')



    result = pd.merge(dfTest, dfTrain, how='left', on=['Country_Region', 'Province_State', 'Year', 'Month', 'Day'])

    result = result.fillna(-1)



    print('Are the Kaggle submission files same? ', np.shape(dfTrain), np.shape(dfTest), np.shape(result))

    result.to_csv('temp.csv')

    return result





# Finding RMSE

def ErrorCalc(mdl, ref, tag):

    relError = np.abs(mdl - ref)/ np.abs(ref+1)

    MeanErrorV = np.mean(relError)

    print(tag + ': Mean Rel Error in %: ', MeanErrorV * 100)

    return MeanErrorV







# Train model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def TrainMdl (trainIpData, trainOpData, tuneTrue) :

    if tuneTrue == 1:

        testSize = 0.5 # 50:50 ratio >> for tuning

    print('Training starts ...')



    #randomState = None # For random train test split

    randomState = 42 # For fixed train test split



    # Split data for final validation

    if tuneTrue == 1:

        X_train, X_test, y_train, y_test = train_test_split(trainIpData, trainOpData, test_size=testSize, random_state=randomState)

        print('Train size: ', np.shape(X_train))

        print('Test size: ', np.shape(X_test))

    else:

        X_train = trainIpData

        y_train =trainOpData

        print('Train size: ', np.shape(X_train))





    # Set 1 - Infected feature dataset

    TrainIP1 = X_train[['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',

                    'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I', 'time_horizon']]

    TrainOP1 = X_train['dayPredictInf']

    if tuneTrue == 1:

        TestIP1 = X_test[['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',

                    'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I', 'time_horizon']]



    if tuneTrue == 1:

        # Model 1 for infection rate prediction

        treeDepth = 30 # Can be changed

        param_grid = {

            'n_estimators': [100, 250, 500],

            'learning_rate': [0.2, 0.1, 0.01]

                    }

        regrMdl1 = xgb.XGBRegressor(max_depth = treeDepth)

        clf1 = RandomizedSearchCV(estimator = regrMdl1, param_distributions = param_grid,

                                         n_iter = 100, cv = 3, verbose=7, random_state=42, n_jobs = -1)

        clf1.fit(TrainIP1, TrainOP1)

        print(clf1.best_estimator_)

        

        # Calculating Error

        y_predictedTrain = clf1.predict(TrainIP1)

        ErrorCalc(y_predictedTrain, X_train['dayPredictInf'].to_numpy(), 'Train Data-set model-1 (infection rate)') # y_predictedTrain converted to numbers



        y_predictedTest = clf1.predict(TestIP1)

        ErrorCalc(y_predictedTest, X_test['dayPredictInf'].to_numpy(), 'Test Data-set model-1 (infection rate)') # y_predictedTest converted to numbers



        

    else:

        print('Directly predicting ...')

        clf1 = xgb.XGBRegressor(max_depth=30, n_estimators = 500, learning_rate = 0.01)

        clf1.fit(TrainIP1, TrainOP1)



    y_predictedTrain = clf1.predict(TrainIP1)

    ErrorCalc(y_predictedTrain, X_train['dayPredictInf'].to_numpy(), 'Train Data-set model-1 (infection rate)') # y_predictedTrain converted to numbers



    



    # Set 2 - Fatality

    TrainIP2 = X_train[['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',

                    'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I',

                    'day1_F', 'day2_F', 'day3_F', 'day4_F', 'day5_F',

                    'diff1_F', 'diff2_F', 'diff3_F', 'diff4_F', 'time_horizon']]

    TrainOP2 = y_train

    if tuneTrue == 1:

        TestIP2 = X_test[['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',

                    'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I',

                    'day1_F', 'day2_F', 'day3_F', 'day4_F', 'day5_F',

                    'diff1_F', 'diff2_F', 'diff3_F', 'diff4_F', 'time_horizon']]





    if tuneTrue == 1:

        # Model 2 for infection rate prediction

        treeDepth = 30  # Can be changed

        param_grid = {

            'n_estimators': [100, 250, 500],

            'learning_rate': [0.2, 0.1, 0.01]

                    }

        regrMdl2 = xgb.XGBRegressor(max_depth = treeDepth)

        clf2 = RandomizedSearchCV(estimator=regrMdl2, param_distributions=param_grid,

                              n_iter=100, cv=3, verbose=7, random_state=42, n_jobs=-1)

        clf2.fit(TrainIP2, TrainOP2)

        print(clf2.best_estimator_)

        

        # Calculating Error

        y_predictedTrain = clf2.predict(TrainIP2)

        ErrorCalc(y_predictedTrain, y_train.to_numpy(), 'Train Data-set model-2')  # y_predictedTrain converted to numbers



        y_predictedTest = clf2.predict(TestIP2)

        ErrorCalc(y_predictedTest, y_test.to_numpy(), 'Test Data-set model-2')  # y_predictedTest converted to numbers

    else:

        print('Directly predicting ...')

        clf2 = xgb.XGBRegressor(max_depth=30, n_estimators = 500, learning_rate = 0.01)

        clf2.fit(TrainIP2, TrainOP2)

    



    # Calculating Error

    y_predictedTrain = clf2.predict(TrainIP2)

    ErrorCalc(y_predictedTrain, y_train.to_numpy(), 'Train Data-set model-2')  # y_predictedTrain converted to numbers



    

    # Validation starts >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    dfValidation_Inf = pd.read_csv('Validation_Infected.csv').reset_index(drop=True)

    dfValidation_Fat = pd.read_csv('Validation_Fatality.csv').reset_index(drop=True)

   



    selRow = 0 # Row count

    startIdx = 20 # To eliminate previous columns



    # Size of the array

    [rVal, cVal] = np.shape(dfValidation_Inf) # Both the validation data-sets are same

    lengthZ = cVal-1

    lengthZ = 12 * lengthZ

    arrP = np.zeros((2 * lengthZ,))

    arrA = np.zeros((2 * lengthZ,))

    count = 0

    error = 0

    errorLen = 0



    print('Validating ...')



    for selRow in range(rVal): # Scans the rows

        #print(count)

        iDetect = startIdx # Starts column scan

        iArray = 0

        # Predict over a horizon

        time_horizon = 0

        while (iDetect < cVal-1): # Scans until the last column : One row one time only

            if iDetect == startIdx : # Only begining count is recorded

                day5_I = dfValidation_Inf.iloc[selRow, iDetect]

                day4_I = dfValidation_Inf.iloc[selRow, iDetect - 1]

                day3_I = dfValidation_Inf.iloc[selRow, iDetect - 2]

                day2_I = dfValidation_Inf.iloc[selRow, iDetect - 3]

                day1_I = dfValidation_Inf.iloc[selRow, iDetect - 4]



                day5_F = dfValidation_Fat.iloc[selRow, iDetect]

                day4_F = dfValidation_Fat.iloc[selRow, iDetect - 1]

                day3_F = dfValidation_Fat.iloc[selRow, iDetect - 2]

                day2_F = dfValidation_Fat.iloc[selRow, iDetect - 3]

                day1_F = dfValidation_Fat.iloc[selRow, iDetect - 4]



                # Run time calculation of other features

                diff1_I = day5_I - day4_I

                diff2_I = day4_I - day3_I

                diff3_I = day3_I - day2_I

                diff4_I = day2_I - day1_I



                diff1_F = day5_F - day4_F

                diff2_F = day4_F - day3_F

                diff3_F = day3_F - day2_F

                diff4_F = day2_F - day1_F



            # Horizon

            time_horizon = time_horizon + 1



            # Infection

            data1 = {'day1_I': [day1_I], 'day2_I': [day2_I], 'day3_I': [day3_I], 'day4_I': [day4_I], 'day5_I': [day5_I],

                     'diff1_I': [diff1_I], 'diff2_I': [diff2_I], 'diff3_I': [diff3_I], 'diff4_I': [diff4_I], 'time_horizon': [time_horizon]}

            dfPredict1 = pd.DataFrame(data1)

            predictedInfected = clf1.predict(dfPredict1)

            if predictedInfected < day5_I:

                predictedInfected = day5_I

            actVal = dfValidation_Inf.iloc[selRow, iDetect + 1]

            arrP[iArray] = predictedInfected

            arrA[iArray] = actVal

            iArray = iArray + 1



            # Fatality

            data2 = {'day1_I': [day1_I], 'day2_I': [day2_I], 'day3_I': [day3_I], 'day4_I': [day4_I], 'day5_I': [day5_I],

                    'diff1_I': [diff1_I], 'diff2_I': [diff2_I], 'diff3_I': [diff3_I], 'diff4_I': [diff4_I],

                    'day1_F': [day1_F], 'day2_F': [day2_F], 'day3_F': [day3_F], 'day4_F': [day4_F], 'day5_F': [day5_F],

                    'diff1_F': [diff1_F], 'diff2_F': [diff2_F], 'diff3_F': [diff3_F], 'diff4_F': [diff4_F], 'time_horizon': [time_horizon]}

            dfPredict2 = pd.DataFrame(data2)

            predictedFatality = clf2.predict(dfPredict2)

            if predictedFatality < day5_F:

                predictedFatality = day5_F

            actVal = dfValidation_Fat.iloc[selRow, iDetect + 1]



            arrP[iArray] = predictedFatality

            arrA[iArray] = actVal

            iArray = iArray + 1





            iDetect = iDetect + 1





        # For each row

        error = error + sum(np.square(np.log(arrP[0:iArray-1] + 1) - np.log(arrA[0:iArray-1] + 1)))



        errorLen = errorLen + iArray #



        #selRow = selRow + 1  # Move to the next row

        count = count + 1





    # Final error

    error = float(error) /  errorLen

    print('Validation error: ', error)





    print('Making Kaggle Submission file ...')



    # Read Kaggle dataset

    dfTrain = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

    dfTest = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

    df2 = kaggle(dfTrain, dfTest) # To pad data





    # Kaggle submission

    print('Prediction starts for Kaggle submission ...')

    [rr, cc] = np.shape(df2)

    time_horizon = 0

    for iP in range(rr):

        if df2.loc[iP, 'ConfirmedCases'] == -1 : # start prediction

            

            time_horizon = time_horizon + 1 # Only time horizon increases



            data1 = {'day1_I': [day1_I], 'day2_I': [day2_I], 'day3_I': [day3_I], 'day4_I': [day4_I], 'day5_I': [day5_I],

                     'diff1_I': [diff1_I], 'diff2_I': [diff2_I], 'diff3_I': [diff3_I], 'diff4_I': [diff4_I], 'time_horizon': [time_horizon]}

            dfPredict1 = pd.DataFrame(data1)

            predictedInfected = clf1.predict(dfPredict1)

            if predictedInfected < df2.loc[iP-1, 'ConfirmedCases']: # Next val must be more than previous val

                predictedInfected = df2.loc[iP-1, 'ConfirmedCases']



            df2.loc[iP, 'ConfirmedCases'] = np.round(predictedInfected)  # Update the current location





            data2 = {'day1_I': [day1_I], 'day2_I': [day2_I], 'day3_I': [day3_I], 'day4_I': [day4_I], 'day5_I': [day5_I],

                    'diff1_I': [diff1_I], 'diff2_I': [diff2_I], 'diff3_I': [diff3_I], 'diff4_I': [diff4_I],

                    'day1_F': [day1_F], 'day2_F': [day2_F], 'day3_F': [day3_F], 'day4_F': [day4_F], 'day5_F': [day5_F],

                    'diff1_F': [diff1_F], 'diff2_F': [diff2_F], 'diff3_F': [diff3_F], 'diff4_F': [diff4_F], 'time_horizon': [time_horizon]}

            dfPredict2 = pd.DataFrame(data2)

            predictedFatality = clf2.predict(dfPredict2)

            if predictedFatality < df2.loc[iP-1, 'Fatalities']: # Next val must be more than previous val

                predictedFatality = df2.loc[iP-1, 'Fatalities']



            df2.loc[iP, 'Fatalities'] = np.round(predictedFatality) # Update the current location



        else:

            

            # Keep capturing values

            if iP > 7 : # Some threshold to eliminate out of index issue

                day5_I = df2.loc[iP-1, 'ConfirmedCases']

                day4_I = df2.loc[iP-2, 'ConfirmedCases']

                day3_I = df2.loc[iP-3, 'ConfirmedCases']

                day2_I = df2.loc[iP-4, 'ConfirmedCases']

                day1_I = df2.loc[iP-5, 'ConfirmedCases']



                day5_F = df2.loc[iP - 1, 'Fatalities']

                day4_F = df2.loc[iP - 2, 'Fatalities']

                day3_F = df2.loc[iP - 3, 'Fatalities']

                day2_F = df2.loc[iP - 4, 'Fatalities']

                day1_F = df2.loc[iP - 5, 'Fatalities']



                # Run time calculation of other features

                diff1_I = day5_I - day4_I

                diff2_I = day4_I - day3_I

                diff3_I = day3_I - day2_I

                diff4_I = day2_I - day1_I



                diff1_F = day5_F - day4_F

                diff2_F = day4_F - day3_F

                diff3_F = day3_F - day2_F

                diff4_F = day2_F - day1_F

            

            time_horizon = 0 # Reset horizon

            

    return df2









# Main code starts >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

print('Size of training data: ', np.shape(df))

trainIpData = df[['day1_I', 'day2_I', 'day3_I', 'day4_I', 'day5_I',

                    'diff1_I', 'diff2_I', 'diff3_I', 'diff4_I',

                    'day1_F', 'day2_F', 'day3_F', 'day4_F', 'day5_F',

                    'diff1_F', 'diff2_F', 'diff3_F', 'diff4_F', 'time_horizon', 'dayPredictInf']]

trainOpData = df['dayPredictFat'] # 'dayPredictInf' is fed to the input set : Just for arrangement

predictions_dF = TrainMdl (trainIpData, trainOpData, tuneTrue) # Kaggle data will be read inside the function

predictions_dF[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv('submission.csv', index=False)