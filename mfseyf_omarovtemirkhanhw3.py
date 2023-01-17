import pandas as pd

import sklearn

import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

import seaborn as sns

%matplotlib inline 
storedf = pd.read_csv("../input/store.csv")

storedf = storedf [["Store", "Assortment", "CompetitionDistance","Promo2"]]

storedf = storedf.set_index("Store")

storedf.CompetitionDistance = storedf.CompetitionDistance.fillna(storedf.CompetitionDistance.max())

storedf.head()
def f(traindf):

    #traindf = traindf[traindf.Open==1]

    traindf = traindf.join(storedf, on="Store")

    traindf['isWeekEnd'] = traindf.DayOfWeek>=5 

    traindf['Month'] = list (map (lambda x: int(x[5:7]), traindf.Date))

    traindf['Day'] = list (map (lambda x: int(x[8:]), traindf.Date))

    traindf['isWinter'] = np.logical_or (traindf.Month <= 2, traindf.Month == 12)

    traindf['isSpring'] = np.logical_and (traindf.Month >= 3, traindf.Month <= 5)

    traindf['isSummer'] = np.logical_and (traindf.Month >= 6, traindf.Month <= 8)

    traindf['isAutumn'] = np.logical_and (traindf.Month >= 9, traindf.Month <= 11)

    traindf['AssortmentA'] = traindf.Assortment=='a'

    traindf['AssortmentB'] = traindf.Assortment=='b'

    traindf['AssortmentC'] = traindf.Assortment=='c'

    traindf['isEndofMonth'] = traindf.Day >= 25

    traindf['isBeginofMonth'] = traindf.Day <= 10

    traindf['CompetitionDistance'] = traindf.CompetitionDistance

    del traindf ["Assortment"]

    del traindf ["StateHoliday"]

    del traindf ["SchoolHoliday"]

    del traindf ["Date"]

    del traindf ["Store"]

    del traindf ["DayOfWeek"]

    del traindf ["Month"]

    del traindf ["Day"]

    return traindf
traindf = pd.read_csv("../input/train.csv", low_memory=False)

traindf = f(traindf)

traindf = traindf[traindf.Open == 1]

ytrain = traindf.Sales.values

del traindf ["Sales"]

del traindf ["Customers"]

del traindf ["Open"]



traindf.head()
testdf = pd.read_csv("../input/test.csv")

testdf = f(testdf)

testdf.head()
model = RandomForestRegressor(min_samples_leaf=2, max_depth=30, n_estimators=30)

%time model.fit(traindf.values, ytrain)
y = model.predict(testdf.values[:, 2:])



df = pd.DataFrame([])

df['Sales'] = y

df['Sales'][testdf.Open == 0] = 0

df = df.set_index(testdf.Id)

pd.DataFrame.to_csv(df, 'ans.csv')

df.head()