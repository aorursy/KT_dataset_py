import pandas as pd

import numpy as np
wine = pd.read_table("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",sep=",",header=None)
wine.head()
wine.drop(wine.columns[[0, 3, 6, 8, 10, 12, 13]], axis = 1, inplace = True)
wine.head()
wine.columns = ["alcohol","malic_acid","alcalinity_of_ash","magnesium","flavanoids","proanthocyanins","hue"]

wine.head()
wine.alcohol.iloc[:3] = np.NAN
wine.head()
wine.magnesium.iloc[2:4] = np.NAN
wine.head()
wine.alcohol = wine.alcohol.fillna(10)

wine.magnesium = wine.magnesium.fillna(100)

wine.head()
x = wine.isnull().sum().sum()

print("Number of mising values in wine dataset is :",x)
randNum = np.random.randint(0,11,10)
randNum
wine.alcohol.iloc[randNum] = np.nan
wine.head(15)
t = wine.alcohol.isnull().sum()

print("Number of mising values in wine dataset is :",t)
y = wine.alcohol.dropna()

y
wine = wine.dropna()

wine
wine.reset_index(drop=True)