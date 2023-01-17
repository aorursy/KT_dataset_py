import numpy as np

import pandas as pd

from pandas import Series, DataFrame
obj = Series([4, 7, -5, 3])
obj.values
obj.index
4 in obj
obj.isnull()
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],        

        'year': [2000, 2001, 2002, 2001, 2002],        

        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}

data
data = DataFrame(data)

data
data.index = ["A","B","C","D","E"]

data
data.drop("A")
data.drop("pop", axis=1)
data[["state", "year"]]
data.loc["A":"C", :]
data.iloc[0:3, :]
data.loc[data["year"] > 2001, ["pop", "state"]]
f = lambda x: x.max() - x.min()

data[["pop", "year"]].apply(f)
f = lambda x: x.max() - x.min()

data[["pop", "year"]].apply(f, axis=1)
obj = Series(["a","b","c","a","a","c","a","b","a","d"])
obj.unique()
obj.value_counts()
from numpy import nan as NA

na_data = DataFrame([

    [1., 6.5, 3.], [1., NA, NA], [NA, NA, NA], [NA, 6.5, 3.]

])

na_data
na_data.dropna()
na_data.dropna(how="all")
na_data.fillna(0)
na_data.fillna(0, inplace=True)

na_data
data.to_pickle("data_pickle")

data = pd.read_pickle("data_pickle")

data