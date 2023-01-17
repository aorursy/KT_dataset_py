testDataDir = "/kaggle/input/home-data-for-ml-course/test.csv"

trainDataDir = "/kaggle/input/home-data-for-ml-course/train.csv"



import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error



# Constant to avoid errors in splitting training data

EPS = 0.0001



# To ignore a recurring FutureWarning in sklearn

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)



# Read in training and testing data

trainFull = pd.read_csv(trainDataDir, index_col='Id')

testFull = pd.read_csv(testDataDir, index_col='Id')



# Drop rows in training data with missing target column

trainFull.dropna(axis=0,subset = ['SalePrice'], inplace = True)



# Separate SalePrice column for prediction

y = trainFull.SalePrice

trainFull.drop(['SalePrice'], axis=1, inplace = True)



# Identify low cardinality categorical variables for one hot encoding

common_cols = [cname for cname in trainFull.columns if cname in testFull.columns]

low_cardinality_cols = [cname for cname in common_cols if trainFull[cname].nunique() < 10 and trainFull[cname].dtype == "object"]

numeric_cols = [cname for cname in common_cols if trainFull[cname].dtype in ['int64', 'float64']]



# Select relevant columns for analysis

my_cols = low_cardinality_cols + numeric_cols

trainData = trainFull[my_cols].copy()

testData = testFull[my_cols].copy()



# One hot encoding of the data

trainData = pd.get_dummies(trainData)

testData = pd.get_dummies(testData)

trainData, testData = trainData.align(testData, join='left', axis = 1)







# Set large number of max estimators for XGB Regressors

MAX_ESTIMATORS = 500



bestTrainSize = 0.85

bestRate = 0.1

lowestError = 20000

for i in range(78,88,1):

    trainSize = i/100

    # Split training data into training and validation set with this training set size

    trainSet, validSet, ytrain, yvalid = train_test_split(trainData, y, train_size= trainSize - EPS, test_size= 1 - trainSize - EPS, random_state=0)



    # Consider different learning rates with this training set size

    for i in range(980,1000,1):

        rate = i/10000

        model = XGBRegressor(n_estimators=MAX_ESTIMATORS,learning_rate=rate, objective = 'reg:squarederror')

        model.fit(trainSet,ytrain,early_stopping_rounds = 40,eval_set=[(validSet,yvalid)], verbose=False)

        predictions = model.predict(validSet)

        mae = mean_absolute_error(predictions,yvalid)

        if mae < lowestError:

            bestRate = rate

            bestTrainSize = trainSize

            lowestError = mae

            print(mae)

print("Best training set ratio:", bestTrainSize)

print("Best learning rate found:",bestRate)

print("Error with this rate:", lowestError)
chosen_model = XGBRegressor(n_estimators = MAX_ESTIMATORS, learning_rate = bestRate, objective = 'reg:squarederror')

chosen_model.fit(trainSet,ytrain, early_stopping_rounds = 40, eval_set=[(validSet,yvalid)], verbose = False)

predictions = chosen_model.predict(testData)

output = pd.DataFrame({'id': testData.index,

                       'saleprice': predictions})

output.to_csv("submission.csv", index=False)