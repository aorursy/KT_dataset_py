# IMPORTING LIBRARIES

import pandas as pd

import numpy as np

import seaborn as sns

import os

import datetime

%matplotlib inline
df = pd.read_csv('../input/hourly-energy-consumption/AEP_hourly.csv')

df.head()

df.info()
df.describe()
#seperate date and time



df["New_Date"] = pd.to_datetime(df["Datetime"]).dt.date

df["New_Time"] = pd.to_datetime(df["Datetime"]).dt.time
df1 = df

df1.head(2)
### When was the higest Energy Consumption and which Year
#Maximum

df1[df1["AEP_MW"] == df["AEP_MW"].max()]
# Mnimum 

df1[df1["AEP_MW"] == df["AEP_MW"].min()]
# Plot and Data visualization



sns.distplot(df1["AEP_MW"])
df1.head(2)
df1["Year"] = pd.DatetimeIndex(df['New_Date']).year
df1.head(2)
df1["Year"].unique()
df1[df1["Year"] == 2013].nunique()
sns.lineplot(x=df1["Year"],y=df1["AEP_MW"], data=df1)
sns.jointplot(x=df1["Year"],

              y=df1["AEP_MW"],

              data=df1,

             kind="reg")
sns.jointplot(x=df1["Year"],

              y=df1["AEP_MW"],

              data=df1,

             kind="kde")
sns.lineplot(x=df1["New_Time"],y=df1["AEP_MW"], data=df1)