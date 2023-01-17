#Add region metadata uploaded by vopani, expand it so that training/testing data has the same information and output to new files



import pandas as pd



def createLocationColumn(dataFrame):

    fillValues = {"Province_State": ""}

    dataFrame = dataFrame.fillna(fillValues)

    location = dataFrame["Province_State"] + dataFrame["Country_Region"]

    dataFrame.insert(0, "Location", location)

    dataFrame = dataFrame.drop(["Province_State", "Country_Region"], axis = 1)

    return dataFrame



def loadAndAppendRegionMetaData(baseData):

    trainingData = createLocationColumn(baseData)

    regionDataBase = pd.read_csv("../input/covid19-forecasting-metadata/region_metadata.csv")

    regionData = createLocationColumn(regionDataBase)

    regionData = regionData.set_index("Location").T

    

    lat = pd.Series([], dtype = regionDataBase.dtypes["lat"])

    lon = pd.Series([], dtype = regionDataBase.dtypes["lon"])

    continent = pd.Series([], dtype = regionDataBase.dtypes["continent"])

    population = pd.Series([], dtype = regionDataBase.dtypes["population"])

    area = pd.Series([], dtype = regionDataBase.dtypes["area"])

    

    latList = []

    lonList = []

    continentList = []

    populationList = []

    areaList = []

    missingSet = set()

    for index, row in trainingData.iterrows():

        try:

            matchedColumn = regionData[row["Location"]]

            latList.append(matchedColumn["lat"])

            lonList.append(matchedColumn["lon"])

            continentList.append(matchedColumn["continent"])

            populationList.append(matchedColumn["population"])

            areaList.append(matchedColumn["area"])

        except KeyError as e:

            missingSet.add(str(e))

            latList.append(0.0)

            lonList.append(0.0)

            continentList.append("")

            populationList.append(0)

            areaList.append(0)

            

    print("missing metadata for:", missingSet)

    lat = pd.Series(latList, dtype = regionDataBase.dtypes["lat"])

    lon = pd.Series(lonList, dtype = regionDataBase.dtypes["lon"])

    continent = pd.Series(continentList, dtype = regionDataBase.dtypes["continent"])

    population = pd.Series(populationList, dtype = regionDataBase.dtypes["population"])

    area = pd.Series(areaList, dtype = regionDataBase.dtypes["area"])

    

    trainingData["lat"] = lat

    trainingData["lon"] = lon

    trainingData["continent"] = continent

    trainingData["population"] = population

    trainingData["area"] = area

    

    print("missing data for these locations:", missingSet)

    return trainingData

    

    

def getTrainingData():

    baseTrainData = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

    Id = baseTrainData["Id"]

    trainDataWithMeta = loadAndAppendRegionMetaData(baseTrainData.drop("Id", axis = 1))

    trainDataWithMeta.insert(0, "Id", Id)

    return trainDataWithMeta



def getTestingData():

    baseTestData = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

    forecastId = baseTestData["ForecastId"]

    testDataWithMeta = loadAndAppendRegionMetaData(baseTestData.drop("ForecastId", axis = 1))

    testDataWithMeta.insert(0, "ForecastId", forecastId)

    return testDataWithMeta







trainingData = getTrainingData()

trainingData.info()

trainingData.to_csv("train_with_metadata.csv", index = False)



testData = getTestingData()

testData.info()

testData.to_csv("test_with_metadata.csv", index = False)

    

    
#This is used to test different algorithms and their effectiveness using a train/test split without having to use limited submissions for feedback



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import sklearn.model_selection as model_selection

#import regressors to test

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.linear_model import LinearRegression

from sklearn.multioutput import MultiOutputRegressor



        

def loadAndFormatPredictors(path):

    baseData = pd.read_csv(path)

    formattedData = pd.get_dummies(baseData, columns = ["Location", "continent"])

    formattedData["Date"] = formattedData["Date"].str.replace("-", "").astype(int)

    

    return formattedData

    



def getFormattedData():

    # load the data

    formattedData = loadAndFormatPredictors("train_with_metadata.csv")

    #remove the id column since it is not useful

    formattedData = formattedData.drop(["Id"], axis = 1)

    return  formattedData.drop(["ConfirmedCases", "Fatalities"], axis = 1),  pd.DataFrame({"ConfirmedCases": formattedData["ConfirmedCases"], "Fatalities": formattedData["Fatalities"]})



predictors, targets = getFormattedData()



scoring = "neg_mean_squared_log_error"



#list of different algorithms we want to test

models = []

models.append(("RF", RandomForestRegressor(2, n_jobs = -1)))

#models.append(("LR", LinearRegression()))

#models.append(('LR', MultiOutputRegressor(LogisticRegression(solver = 'saga', multi_class = 'ovr', verbose=1, n_jobs=-1)))) # takes forever

#models.append(("SVM", SVC(gamma="auto", verbose = 1))) # also takes forever



#evaluate each model in turn

resultMessages = []

for name, model in models:

    print("Evaluating", name)

    kfold = model_selection.KFold(n_splits = 5)

    

    #evaluate model

    cvResults = model_selection.cross_val_score(model, predictors, targets, cv = kfold, scoring = scoring)

    msg = "%s: %f (%f)" % (name, cvResults.mean(), cvResults.std())

    resultMessages.append(msg)

    

print("Prediction Results:")

for result in resultMessages:

    print(result)



    

    
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

        

def loadAndFormatPredictors(path):

    baseData = pd.read_csv(path)

    formattedData = pd.get_dummies(baseData, columns = ["Location", "continent"])

    formattedData["Date"] = formattedData["Date"].str.replace("-", "").astype(int)

    

    return formattedData

    

def getFormattedTrainingData():

    # load the data

    formattedData = loadAndFormatPredictors("train_with_metadata.csv")

    #remove the id column since it is not useful

    formattedData = formattedData.drop(["Id"], axis = 1)

    return  formattedData.drop(["ConfirmedCases", "Fatalities"], axis = 1),  pd.DataFrame({"ConfirmedCases": formattedData["ConfirmedCases"], "Fatalities": formattedData["Fatalities"]})



def getFormattedTestingData():

    formattedData = loadAndFormatPredictors("test_with_metadata.csv")

    return formattedData

    

trainingPredictors, trainingTarget = getFormattedTrainingData()



testingData = getFormattedTestingData()



from sklearn.ensemble import RandomForestRegressor

from sklearn.multioutput import MultiOutputRegressor



rfg = RandomForestRegressor(300, n_jobs = -1)



print("training")

rfg.fit(trainingPredictors, trainingTarget)

print("finished training")



predictions = rfg.predict(testingData.drop("ForecastId", axis = 1))

        

finalDataFrame = pd.DataFrame({"ForecastId": testingData["ForecastId"], "ConfirmedCases": predictions[:, 0], "Fatalities": predictions[:, 1]})

finalDataFrame.info()

finalDataFrame.to_csv("submission.csv", index = False)