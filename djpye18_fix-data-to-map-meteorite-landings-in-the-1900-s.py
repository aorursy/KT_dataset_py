import numpy as np

import pandas as pd

import csv
fileSource = pd.DataFrame(pd.read_csv("meteorite-landings.csv"))

fileSource.head()
dataSet = fileSource.drop(['id','nametype','recclass','fall','GeoLocation'], axis = 1)
dataSet = dataSet.sort_values(by=['year', 'name'], ascending=[True, True])

# print(fileEdited.head())
# dropping all N/A 

dataSet = dataSet.dropna()
# criteria 1

criteria1 = dataSet["year"].map(lambda x: int(x) >= 1900)

dataSet = dataSet[criteria1]

                                

# criteria 2

criteria2 = dataSet["year"].map(lambda x: int(x) < 2000)

dataSet = dataSet[criteria2] 
dataSet.describe()
# criteria 2

criteria3 = dataSet["reclat"].map(lambda x: np.float64(x)  != np.float64(0))

dataSet = dataSet[criteria3]
dataSet.to_csv('1900sMeteoriteHits.csv')