import pandas as pd
import numpy as np
import os

list =  ["Brazil", "Russia", "India", "China", "South Africa"]
places = pd.Series(list)
places.head()
places = pd.Series(list,index=["BR", "RU", "IN", "CH", "SA"])
places.head()
# places.index = ["BR", "RU", "IN", "CH", "SA"]
dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }

info = pd.DataFrame(dict) #change index
info.head()
df_from_series = pd.DataFrame(places,columns=["a"]) #df_from_series.columns
df_from_series.head()
print(os.listdir("../input"))

df = pd.read_csv("../input/pokemon.csv") # header=None
df.head(10)
df.shape
df.columns

df[['Name','Speed']]
del(df['Form'])
df.head()
#slicing
df[0:4]
#loc vs iloc
df.iloc[:2,:4]
df.loc[:2]
df.T
df.axes
df.dtypes
df.empty
df.ndim
df.shape
df.size
df.values
df.head()
df.tail(100)
df.sum()

df.mean()
df.corr()
df.count()
df.max()
df.min()
df.median()
df.std()
df.describe() # 25%,50%,35%  min+(max-min)*percentile 
df = pd.DataFrame([0,1,2,3],index = ['a','b','c','d'])
df
df.reindex(columns = [0,1,2,3])
df.reindex(columns = [0,1,2,3],fill_value = "ml class")
df.reindex(index = ['a','c','d'])


