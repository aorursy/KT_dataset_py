import numpy as np

import pandas as pd

import seaborn as sns
data = pd.read_csv("../input/weather-dataset-rattle-package/weatherAUS.csv")

data.drop( labels=["Date", "Evaporation", "Sunshine"], axis=1 )
minTemp = data[

    ["Location", "MinTemp"]

].groupby(by="Location").min()

minTemp = minTemp["MinTemp"].sort_values()[::-1]
print(minTemp)
sns.barplot( minTemp, minTemp.index )