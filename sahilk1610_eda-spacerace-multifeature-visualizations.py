import numpy as np # linear algebra
import datetime
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from collections import Counter
file = pd.read_csv("../input/all-space-missions-from-1957/Space_Corrected.csv")
file.head()
def trim(s):
    s = s.replace(",", "")
    return s[ 3: -3] 
file["Datum"] = file["Datum"].apply(lambda x : trim(x))
#helper funciton to add certain new variables from our date column

def add_datepart(df, field_name, prefix=None, drop=True, time=False):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    attr = ['year', 'month', 'week', 'day', 'hour', 'minute', 'second', 'is_month_start']
    attr_deprecated = ['Week']
  
    for n in attr:
        if n not in attr_deprecated: df[prefix + n] = getattr(pd.DatetimeIndex(df[field_name]), n)
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df
file = add_datepart(file, "Datum",prefix = "date_", time = True)
plt.rcParams["figure.figsize"] = 16 , 12
df = file.groupby(["date_year", "Status Mission"]).agg({"Status Mission": "count"}).unstack()
df.plot(kind = "bar", stacked= True)

plt.show()
plt.rcParams["figure.figsize"] = 12 , 8
df2 = file.groupby(["date_month", "Status Mission"]).agg({"date_month": "count"}).unstack()
df2.plot(kind = "bar", stacked = True)
plt.show()
plt.rcParams["figure.figsize"] = 12 , 8
df5 = file.groupby(["date_day", "Status Mission"]).agg({"date_day": "count"}).unstack()
df5.plot(kind = "bar", stacked = True)
plt.show()
plt.rcParams["figure.figsize"] = 16 , 12
df3 = file.groupby(["date_year", "Status Rocket"]).agg({"Status Rocket": "count"}).unstack()
df3.plot(kind = "bar", stacked= True)
plt.title(" Rocket status according to the Year")
plt.show()
df4 = file.groupby(["Company Name", "Status Rocket"]).agg({"Status Rocket": "count"}).unstack()
df4.plot(kind = "bar", stacked = True)
plt.show()
file['Location_Country'] = file['Location'].apply(lambda x: x.split(',')[-1])
df5 = file.groupby(["Location_Country", "Status Mission"]).agg({"Status Mission": "count"}).unstack()
df5.plot(kind = "bar", stacked = True)
plt.show()
