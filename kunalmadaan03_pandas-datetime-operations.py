import pandas as pd

import numpy as np

import datetime as dt
pd.read_csv("../input/mydataset/wind.data")
data = pd.read_csv("../input/mydataset/wind.data")

data["Date"] = pd.to_datetime(data[["Yr","Mo","Dy"]].astype(str).agg('-'.join, axis=1))

data = data.drop(columns=["Yr","Mo","Dy"])

data.head()
data["Date"] = np.where(pd.DatetimeIndex(data["Date"]).year < 2000,data.Date,data.Date - pd.offsets.DateOffset(years=100))
newData = data.set_index("Date")

newData.index.astype("datetime64[ns]")
print(newData.isnull().values.ravel().sum())
x=newData.count()

print("Total Non-missing values are :",x.sum())
y = newData.mean()

y.mean()
def stats(x):

    x = pd.Series(x)

    Min = x.min()

    Max = x.max()

    Mean = x.mean()

    Std = x.std()

    res = [Min,Max,Mean,Std]

    indx = ["Min","Max","Mean","Std"]

    res = pd.Series(res,index=indx)

    return res

loc_stats = newData.apply(stats)

loc_stats
day_stats = newData.apply(stats,axis=1)

day_stats.head()
january_data = newData[newData.index.month == 1]

print ("January windspeeds:")

print (january_data.mean())
print( "Yearly:\n", newData.resample('A').mean())
print ("Monthly:", newData.resample('M').mean())
print ("Weekly:", newData.resample('W').mean())
newdata = newData.groupby(lambda d: (d.month, d.year))

print ("Mean wind speed for each month in each location")

print (newdata.mean())
first_year = newData[newData.index.year == 1961]

stats1 = newData.resample('W').mean().apply(lambda x: x.describe())

print (stats1)