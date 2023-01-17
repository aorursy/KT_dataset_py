# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

wifi = pd.read_csv("../input/UJIData.csv", index_col=0)



wifi.head(5)

# Function definitions

def EucDist(x1, y1, x2, y2):

    return np.sqrt( np.power((x1-x2),2) + np.power((y1-y2), 2) )

#looking at the features



wifi.describe()



#creating dataframe



wifi=pd.DataFrame(wifi)

print(wifi)
#remove the other 'target variables' for Longitude

Long = wifi.drop(wifi.columns[[0,521,522,523,524,525,526,527,528,529]], axis=1)

Long.head(5)
#remove the other 'target variables' for Latitude

Lat = wifi.drop(wifi.columns[[0,520,522,523,524,525,526,527,528,529]], axis=1)

Lat.head(5)
ds_ = pd.read_csv('../input/UJIData.csv')

ds = ds_.drop(ds_.columns[0], axis=1)



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



wifiDf = ds.iloc[:, 1:520]

wifiDf.replace(to_replace=100, value=-200)

#print(wifiDf)

lats = ds['LATITUDE']

#print(lats)

lons = ds['LONGITUDE']

#lons



xTrain, xTest, latTrain, latTest = train_test_split(wifiDf, lats, test_size=0.3, random_state=2)

xTrain, xTest, lonTrain, lonTest = train_test_split(wifiDf, lons, test_size=0.3, random_state=2)



rfLat = RandomForestRegressor(n_estimators=500,

                              min_samples_leaf=20,

                              max_features="sqrt",

                              oob_score=True,

                              n_jobs=-1,

                              verbose=1)



rfLat.fit(xTrain, latTrain)

predLat = rfLat.predict(xTest)

#print(predLat)

#print(latTest)
print(predLat)# Use the forest's predict method on the test data

#predictions = rf.predict(test_features)

# Calculate the absolute errors

errors = abs(predLat - latTest)

print(errors)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

print( "MAE = " + str(np.mean(np.absolute(predLat - latTest))) )



rfLon = RandomForestRegressor(n_estimators=500,

                              min_samples_leaf=20,

                              max_features="sqrt",

                              oob_score=True,

                              n_jobs=-1,

                              verbose=1)



rfLon.fit(xTrain, lonTrain)

predLong = rfLon.predict(xTest)
print(predLong)# Use the forest's predict method on the test data

#predictions = rf.predict(test_features)

# Calculate the absolute errors

errors = abs(predLong - lonTest)

print(errors)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

#print( "MAE = " + str(np.mean(np.absolute(predLat - latTest))) )
dists = EucDist(predLat, predLong, latTest, lonTest)

meanED = np.mean(dists)

maxED = np.max(dists)

minED = np.min(dists)

print("--Metrics for Random Forests UJI Data--")

print("meanED = " + str(meanED) + " m")

print("maxED = " + str(maxED) + " m")

print("minED = " + str(minED) + " m")