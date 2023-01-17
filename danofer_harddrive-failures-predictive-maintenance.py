import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/harddrive.csv')
print(df.shape)
df.head()
# drop constant columns
df = df.loc[:, ~df.isnull().all()]
print(df.shape)
# number of hdd
print("number of hdd:", df['serial_number'].value_counts().shape) 

# number of different types of harddrives
print("number of different harddrives", df['model'].value_counts().shape)
failed_hdds = df.loc[df.failure==1]["serial_number"]
len(failed_hdds)
df = df.loc[df["serial_number"].isin(failed_hdds)]
df.shape
df["end_date"] = df.groupby("serial_number")["date"].transform("max")
df.tail()
df["end_date"] = pd.to_datetime(df["end_date"])
df["date"] = pd.to_datetime(df["date"])
df["date_diff"] = df["end_date"] - df["date"]
df["date_diff"].describe()
# replace string/object with number
df["date_diff"] = df["date_diff"].dt.days
df.drop(["end_date","failure"],axis=1,inplace=True)
print(df.shape)
df.to_csv("smartHDD_Failures_2016_survival.csv.gz",index=False,compression="gzip")
print("done")
