

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
res_levels=pd.read_csv("../input/chennai-water-management/chennai_reservoir_levels.csv")

res_rainfall=pd.read_csv("../input/chennai-water-management/chennai_reservoir_rainfall.csv")
res_levels.head(3)
res_rainfall.head(5)
res_rainfall.isna().sum()
res_levels.isna().sum()
res_levels.dtypes
res_rainfall.dtypes
df=res_rainfall.merge(res_levels,on='Date',how="inner",suffixes=("_rain","_res"))
df.head(4)
res_rainfall.Date=pd.to_datetime(res_rainfall.Date,format="%d-%m-%Y")

res_rainfall.dtypes

res_rainfall.index=res_rainfall.Date

res_rainfall=res_rainfall.drop(['Date'],axis=1)
plt.figure(figsize=(12,5))

plt.plot(res_rainfall.POONDI)

plt.title("Rain Fall At POONDI")

plt.xlabel("year")

plt.ylabel("rain fall in Cm")

plt.figure(figsize=(12,5))

plt.plot(res_rainfall.POONDI,color='r',label='Poondi')

plt.plot(res_rainfall.CHOLAVARAM,color='y',label='CHOLAVARAM')

plt.plot(res_rainfall.REDHILLS,color='b',label='REDHILLS')

plt.plot(res_rainfall.CHEMBARAMBAKKAM,color='g',label='CHEMBARAMBAKKAM')

plt.legend()

plt.title("Rain Fall At POONDI")

plt.xlabel("year")

plt.ylabel("rain fall in Cm")
res_rainfall.head()
res_rainfall["Total"]=(res_rainfall.POONDI+res_rainfall.CHOLAVARAM+res_rainfall.REDHILLS+res_rainfall.CHEMBARAMBAKKAM)/4
res_rainfall
plt.figure(figsize=(12,5))

plt.plot(res_rainfall.Total,color='r')

plt.title("avg rainfall")

plt.xlabel("year")

plt.ylabel("rain in Cm")
