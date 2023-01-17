import numpy as np

import pandas as pd

from pandas import DataFrame
df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],'data1': range(7)})

df2 = DataFrame({'key': ['a', 'b', 'd'],'data2': range(3)})
df1
df2
pd.merge(df1, df2)
pd.merge(df1, df2, on='key')
pd.merge(df1, df2, left_on='key', right_on="key")
pd.merge(df1, df2, how="outer")

# outer: keep both NaN

# inner:  skip NaN from both

# left: skip NaN from right

# right: skip NaN from left
arr = np.arange(12).reshape(3,4)
np.concatenate([arr, arr])
np.concatenate([arr, arr], axis=1)
data = [

    ["A", "A", "B", "C", "D"],

    [1,2,3,4,5],

    ["X1", "X2", "X3", "X4", "X5"]

]

data = np.array(data)

data = data.T

data = np.concatenate([data,data,data,data, data,data,data,data])

data = DataFrame(data)

data.columns = ["C1", "V1", "C2"]

data.head()
pd.pivot_table(data, values='V1', index=['C1'], columns=['C2'], aggfunc=np.max)
data.drop_duplicates()
to_value = {"X1":1, "X2": 2, "X3": 3, "X4": 4, "X5":5}

data["C2"] = data["C2"].map(to_value)

data.head()