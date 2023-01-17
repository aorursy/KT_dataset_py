import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import datetime
trainData = pd.read_csv('../input/train.csv')
testData = pd.read_csv('../input/test.csv')
trainData.describe()
#check if any empty value
trainData.isnull().sum()
def timeFeatures(df):
    df['datetime'] = pd.to_datetime(df['date'])
    df['dow']      = df['datetime'].dt.dayofweek
    df["doy"]      = df["datetime"].dt.dayofyear
    df["year"]     = df['datetime'].dt.year
    df["month"]    = df['datetime'].dt.month
    df.drop(['datetime'], axis=1, inplace=True)
    return df
trainData = timeFeatures(trainData)
trainData.head()
#visualization part
testData = timeFeatures(testData)
testData.head()
trainData.drop(['date'], axis=1, inplace=True)
testData.drop(['date'], axis=1, inplace=True)
yTrain = trainData['sales']
trainData.drop(['sales'], axis=1, inplace=True)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 50)
model.fit(trainData,yTrain)
model.score(trainData,yTrain)
#cleaning and modifying test data
yTest = testData['id']
testData.drop(['id'],axis=1, inplace=True)
pred = model.predict(testData)
my_submission = pd.DataFrame({'id': yTest, 'sales': pred})
my_submission.to_csv('submission.csv', index=False)
