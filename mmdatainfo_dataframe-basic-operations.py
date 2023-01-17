import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import scipy as sp
from sklearn.impute import SimpleImputer
# Vector input for each column

df1 = pd.DataFrame({"a": [i for i in range(1,4)], "b": [10,20,30]})

df1.head()
# Just like in the case above but providing index.

df1 = pd.DataFrame({"a": [i for i in range(1,4)], "b": [10,20,30]},index=range(1,4))

df1.head()
# df1.index will still be RangeIndex!! but with start = 1 (not 0 as for the first case)

df1.index
# Matrix/Array with all data and column name vector

df2 = pd.DataFrame([[1,"one"],[2,"two"]],columns=["a","c"])

# Same as 

#df2 = pd.DataFrame(np.array([[1,"one"],[2,"two"]]),columns=["a","c"])

#

# could also use tuple inside array

#df2 = pd.DataFrame([(1,"one"),(2,"two")],columns=["a","c"])

df2.head()
# !readlink -f ../input/
# Read directly to DataFrame

df4 = pd.read_csv("../input/test-data/test_csv.csv")

df4.head()
# Set new index after creating dataframe

dfindex = pd.DataFrame({"a": [i for i in range(1,4)], "b": [10,20,30]})
# RangeIndex by default

dfindex.index
# Will stay RangeIndex if `inplace`

dfindex.set_index("a")

dfindex.index # was not overwritten
dfindex["b"][0]
# Replace "inplace"

dfindex.set_index("a",inplace=True)

dfindex.head()
dfindex.index
dfindex["b"][1]

# same as 

dfindex["b"].loc[1]
# `iloc` function can be use to employ 0 to length-1 indexing regardless of `index`

dfindex["b"].iloc[0]
# use return a DataFrame (not a pd.Series) use []

dfindex.iloc[[0]]
# Return one column = pd.Series (all rows)

df1["a"]

# same as

df1.a
# Return mupltiple columns (all rows). Here just like in case of indices, use [[]] to return pd.Series (not pd.Series)

df1[["a","b"]]
# Return all except "b" (do not use inplace to keep in original)

df1.drop(["b"],axis=1)
# Return first row

df1.iloc[0] # in this case same as df1.loc[0]
# show second element/column (="b") of the first row

df1.iloc[0,1]
# Return b column, first and last row. Use the `loc` function to access index for multiple entries

# the -1 index does not work with `loc`

df1.b.loc[[0,df1.shape[0]-1]]
# Use conditions returning all columns

df1[df1.b>=20]
# For multiple conditions, use tuple

df1[(df1.b>=20) & (df1.a<3)]
# Same as above but allowing to return selected columms only

df1[["a","b"]].loc[df1.b>=20]
# get row index for condition

df1[df1.b>=20].index
# Apply built-in functions to one column

df1.a.mean()

# Same when using external function

np.mean(df1.a)
# Apply to all columns

df1.mean()

# Same as 

df1.mean(axis=0)

# Same as 

df1.apply(np.mean,axis=0)
# Apply to all rows

df1.mean(axis=1)

# Same as 

df1.apply(np.mean,axis=1)

# Same as (=transpose the dataframe and compute over axis=0)

np.mean(df1.T)
# Apply to all columns and rows

df1.values.mean()

# Same as

np.mean(df1.values)
# show prior modification

df1
# use mask function to modify all values that meet criteria in one column (inplace)

df1.a.mask((df1.a>0) & (df1.b==20),200,inplace=True)

df1
# change all values > 25 (inplace)

df1[df1>25] = df1[df1>25]*100

df1
# use update function to modify values using given pd.Series (or DataFrame)

# update is applied inplace!!

x = pd.Series([999],name="b",index=[3])

df1.update(x)

df1
# Show all NaNs

df4.isna()

# same as 

df4.isnull()
# Show only the total number of NaNs per column

df4.isnull().sum()
# Remove rows with NaNs. Can use `inplace` to overwrite the DataFrame insteade returning a copy without NaNs

df4.dropna()
# Replace NaNs with 0

df4.replace({np.NaN:0})

# same as 

df4.fillna(0)
# Replace with mean value in the second column inplace

df4.col2.replace({np.NaN:df4.col2.mean()},inplace=True)
df4.head()
# Interpolate column 1 using index as x coordinate

df4.col1.interpolate(inplace=True)
df4.head()
df4.col3.interpolate()
# Use sklearn impute library

imp = SimpleImputer(strategy="most_frequent",missing_values=np.NaN)

# Need a matrix copy (use [] even for one column)

X = df4.loc[:,["col3"]].copy()

# Show result

imp.fit_transform(X)
# DataFrame with daily values `[D]` = datetime & string vector (will be converted)

df_time = pd.DataFrame({"datetime":np.arange("2010-01-01", "2010-01-05", dtype="datetime64[D]"),

                        "datestring":["2010/01/01","2010/02/01","NULL","2010/04/01"]})

df_time
df_time.dtypes
df_time.datetime>np.datetime64("2010-01-03")
pd.to_datetime(df_time.datestring,format="%Y/%d/%m",errors="coerce")
df_type = pd.DataFrame({"b":[True,True,False,True],

                        "i":[1,2,3,4],

                        "f":[1.2,2.6,np.NaN,4.0],

                        "s":["one","two","three","four"],

                        "d":np.arange("2010-01-01", "2010-01-05", dtype="datetime64[D]")})
df_type.dtypes
df_copy = df_type.copy();

df_copy.loc[2,"i"] = np.NaN

df_copy.dtypes
df_type.f.replace({np.NaN:0}).astype("int") # will not be converted "inplace"
# Append new column to a copy. Does not support inplace

df4.assign(col4=["a","b","c","d"])
# was not appended to actual dataframe but to a copy

df4.head(2)
# Append to existing dataframe = just as using inplace

df4 = df4.assign(col4=["a","b","c","d"])

# Same as 

df4["col4"] = ["a","b","c","d"]

df4.head()
# Append new row. use ignore_index to continue index of original DataFrame

df4.append(pd.DataFrame([[1001,12,np.NaN,"e"]],columns=df4.columns),ignore_index=True).head()
# Same as horizontal concatenation. For vertical (add column) use "axis=1"

pd.concat([df4,pd.DataFrame([[1001,12,np.NaN,"e"]],columns=df4.columns)])
# Just show df1

df1.head()
# Create new DataFrame that will be merged with df1. On purpose, use different index

df2 = pd.DataFrame({"c":[100,200,300]},index=[1,2,3])

df2.head()
# Merge on index. Use full outer join (inner by default)

dfmerge = df1.merge(df2,left_index=True,right_index=True,how="outer")

dfmerge.head()
# Create DataFrame for groupby function

df = pd.DataFrame({"a":np.random.randn(10),"b":["a","a","b","c","c","c","e","e","e","e"]})

df.head()
# Compute mean for each group

df.groupby("b").mean()

# df.a[df.b=="b"].mean()
# groupby creates new object

g = df.groupby("b")

# Could itereate over "groups"

g.groups
# Compute mean only group "c" (compare to result above=is like index)

g.get_group("c").mean()