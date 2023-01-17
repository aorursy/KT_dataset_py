import pandas as pd
import numpy as np
#create empty series
df = pd.Series()
df.shape
#create series from ndarray
array = np.array(np.random.rand(10))
df_from_array = pd.Series(array)
df_from_array
dict = {"name":"Abhishek","marks":70}
df_from_dict = pd.Series(dict)
df_from_dict
df = pd.Series(['a','b','c','d'],index = [0,1,2,3])
df
df[0]
df[:3]
df[df == 'a']
df[[3, 2, 1]] # passing list for accessing multiple objects
df = pd.DataFrame(np.random.randn(5,3),index = [0,2,4,5,7],columns = ['a','b','c'])
df
df = df.reindex(index = [0,1,2,3,4,5,6,7],columns = ['a','b','c'])
df
df.info()
df.describe()
df.isnull().sum()
df.fillna(0)
df.replace(0, np.nan, inplace= True)
df
df.fillna(df.mean())
df=df.dropna()
df
