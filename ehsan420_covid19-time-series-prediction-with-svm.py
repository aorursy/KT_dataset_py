import numpy as np 

import pandas as pd

from copy import deepcopy

from sklearn.metrics import mean_absolute_error

%matplotlib inline 

random_state=0
# The data is provided by the Johns Hopkins University

# https://github.com/CSSEGISandData/COVID-19

df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df.head()
# only work with a limited amount of data

df = df[df.columns[:82]]
# rawData contains the values of confirmed cases in the world

firstDateIndex = df.columns.get_loc("1/22/20")

rawData = [0 for i in range(firstDateIndex, df.shape[1])]

for i in range(df.shape[1] - firstDateIndex):

    rawData[i] = sum(df.iloc[:, i+firstDateIndex])
import matplotlib.pyplot as plt 

plt.figure(figsize=(10, 5))

plt.title('Number of confirmed Coronavirus cases in the world', size=20)

plt.plot(range(len(rawData)), rawData)

plt.show()
import matplotlib.pyplot as plt 

plt.figure(figsize=(10, 5))

plt.title('Number of confirmed Coronavirus cases in the world', size=20)

plt.plot(range(len(rawData[50:])), rawData[50:])

plt.show()
originalData = deepcopy(rawData)

rawData = rawData[50:]
numberOfDaysToForcast = 5
import datetime

numberOfAvailableDays = len(rawData)

numberOfDays = numberOfAvailableDays + numberOfDaysToForcast

day1 = datetime.datetime.strptime('1/22/2020', '%m/%d/%Y')

day50 = day1 + datetime.timedelta(days=50)

availableDateRange = [day50 + datetime.timedelta(days=x) for x in range(numberOfAvailableDays)]

futureDateRange = [availableDateRange[-1]  + datetime.timedelta(days=x) for x in range(1, numberOfDaysToForcast + 1)]

fullAvailableDateRange = [day1 + datetime.timedelta(days=x) for x in range(len(originalData))]

print('Dates to be predicted:')

futureDateRange
# This is how I form my dataset from the rawData variable



# rawData = [1 2 3 4 5]:

# x=[1 2] y=3

# x=[2 3] y=4

# x=[3 4] y=5

# number of training data = 5 - 2



# rawData = [1 2 3 4 5 6]:

# x=[1 2] y=3

# x=[2 3] y=4

# x=[3 4] y=5

# x=[4 5] y=6

# number of training data = 6 - 2
numberOfPreviousData = 7
testFraction = 0.3

numberOfTrainingTestRecords = len(rawData) - numberOfPreviousData

numberOfUsableData = numberOfTrainingTestRecords - numberOfPreviousData + 2

numberOfTestData = int(np.ceil(testFraction * numberOfUsableData))
# Select a good machine learning algorithm for this dataset By picking the one that fits the data well and has the lowest error
y_actual = np.array(rawData[-numberOfTestData:])

y_predicted = np.array(rawData[-numberOfTestData-1:-1])

errors = np.abs(y_actual - y_predicted) / np.array(y_predicted)

baselineError = errors.mean()

print('MAE error: ', baselineError)
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor
def findBestParameters(model, train_X, train_y):

    modelName = model.__class__.__name__

    parameters = {}

    if modelName == 'SVR':

        parameters = {

            'C':[1000, 100, 10, 1],

            'epsilon': [0.1, 0.01, 0.05, 0.001],

            'gamma': ['scale', 'auto']}

        

    elif modelName == 'RandomForestRegressor':

        parameters = {

            'n_estimators': [10, 50, 100, 150, 200, 400, 600]}

        

    elif modelName == 'MLPRegressor':

        parameters = {

            'hidden_layer_sizes': [(5), (10), (20), 

                        (5, 5), (10, 10), (20, 20), (30, 30),

                        (5, 5, 5), (10, 10, 10), (20, 20, 20)],

            'max_iter': [100, 500, 1000, 2000]}

    

    clf = GridSearchCV(model, param_grid=parameters, cv=5)

    clf.fit(train_X, train_y)

    #print('Best parameters for', modelName, '\n', clf.best_params_)

    

    if modelName == 'SVR':

        return SVR(C=clf.best_params_['C'], epsilon=clf.best_params_['epsilon'], gamma=clf.best_params_['gamma'])

    elif modelName == 'RandomForestRegressor':

        return RandomForestRegressor(n_estimators=clf.best_params_['n_estimators'])

    elif modelName == 'MLPRegressor':

        return MLPRegressor(hidden_layer_sizes=clf.best_params_['hidden_layer_sizes'], max_iter= clf.best_params_['max_iter'])

    return np.nan
# Build the dataset for training and testing

X = np.zeros((numberOfTrainingTestRecords, numberOfPreviousData))

y = np.zeros(numberOfTrainingTestRecords)

for i in range(numberOfTrainingTestRecords):

    for j in range(numberOfPreviousData):

        X[i, j] = deepcopy(rawData[i + j])

    y[i] = deepcopy(rawData[i + numberOfPreviousData])
# Split the dataset into train and test

X_train = X[0: -numberOfTestData, :]

y_train = y[0: -numberOfTestData]

X_test = X[-numberOfTestData:, :]

y_test = y[-numberOfTestData:]
# Normalize

maxValue = max(rawData)

X_train = X_train /maxValue

y_train = y_train /maxValue

X_test = X_test /maxValue

y_test = y_test /maxValue
def recursiveForcast(model):

    errors = np.zeros(numberOfDaysToForcast)

    dataUsedToForcast = np.concatenate([y_train[-numberOfPreviousData:], y_test]).copy()

    for f in range(len(y_test) - numberOfDaysToForcast + 1):

        movingX = deepcopy(dataUsedToForcast[f: f + numberOfPreviousData])

        for i in range(numberOfDaysToForcast):

            newForcast = model.predict(movingX.reshape(1, -1))

            observedValue = dataUsedToForcast[f + numberOfPreviousData + i]

            # define the error to be the fraction of observed value

            currentError = np.abs(newForcast - observedValue)

            errors[i] = errors[i] + currentError/newForcast

            for j in range(movingX.shape[0] - 1): # Shift

                movingX[j] = movingX[j+1]

            movingX[-1] = newForcast

    recursiveMethodError = errors / (len(y_test) - numberOfDaysToForcast + 1)

    print('MAE error for', model.__class__.__name__, recursiveMethodError)

    return recursiveMethodError
allModels = [SVR(), MLPRegressor(), RandomForestRegressor()]

recursiveError = []

for m in allModels:

    m = findBestParameters(m, X_train, y_train)

    m.fit(X_train, y_train)

    recursiveMethodError = recursiveForcast(m)

    if m.__class__.__name__ == 'SVR':

        recursiveError = recursiveMethodError
# train the SVM model on all available data and then make predictions for the upcoming days
# Build the dataset for training and testing

X = np.zeros((numberOfTrainingTestRecords, numberOfPreviousData))

y = np.zeros(numberOfTrainingTestRecords)

for i in range(numberOfTrainingTestRecords):

    for j in range(numberOfPreviousData):

        X[i, j] = deepcopy(rawData[i + j])

    y[i] = deepcopy(rawData[i + numberOfPreviousData])
# Split the dataset into train and test

X_train = X.copy()

y_train = y.copy()

X_test = deepcopy(np.array(rawData[-numberOfPreviousData:]))
# Normalize

maxValue = max(rawData)

X_train = X_train /maxValue

y_train = y_train /maxValue

X_test = X_test /maxValue
def recursiveForcast(model):

    movingX = deepcopy(X_test)

    predictions = np.zeros(numberOfDaysToForcast)

    for i in range(numberOfDaysToForcast):

        newForcast = model.predict(movingX.reshape(1, -1))

        for j in range(movingX.shape[0] - 1): # Shift

            movingX[j] = movingX[j+1]

        movingX[-1] = newForcast

        predictions[i] = newForcast

    predictions = predictions * maxValue

    predictions = [int(np.round(x)) for x in predictions]

    print('Predictions:', predictions)

    return predictions
model = SVR()

model = findBestParameters(model, X_train, y_train)

model.fit(X_train, y_train)

recursivePredictions = recursiveForcast(model)
plt.figure(figsize=(20, 10))

plt.rcParams['xtick.labelsize'] = 20 

plt.rcParams['ytick.labelsize'] = 20 

plt.plot(fullAvailableDateRange, originalData)

plt.plot(futureDateRange, recursivePredictions, linestyle='dashed', color='red')

lowerBound = recursivePredictions - recursiveError * recursivePredictions

upperBound = recursivePredictions + recursiveError * recursivePredictions

plt.fill_between(futureDateRange, lowerBound, upperBound, color='b', alpha=.1)

plt.legend(['Real values', 'Predictions'], prop={'size': 20}, loc=2)

plt.xticks( rotation=90)

x1,x2,y1,y2 = plt.axis()

plt.axis((x1,x2,y1,maxValue*2))

plt.show()
print('Number of confirmed cases:')

for i, day in enumerate(futureDateRange):

    print(day.date(), ':', recursivePredictions[i], 'error:', int(round(recursiveError[i]*maxValue)))