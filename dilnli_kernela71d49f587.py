import numpy as np
import pandas as pd
import pandas_profiling as PR
import math
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
import os

df = pd.read_csv("../input/911.csv")

df.head() #
df.info()
df.describe()
type(df["timeStamp"][0])
# df["zip"].unique() #уникальные значения
# df["zip"].value_counts().head(10) #список zip 10 первыйх(больших)
# df["title"].value_counts()
def word(title):
    w = title.split(":")
    return w[0]
word
df["Reason"] = df["title"].apply(word)
# df["Reason"].unique()

df["timeStamp"] = pd.to_datetime(df["timeStamp"])
df["timeStamp"].head()
time = df["timeStamp"].iloc[1]
time.hour
time.month
time.dayofweek
time.year
time.minute
time.second
df["Hour"] = df["timeStamp"].apply(lambda t:t.hour) # Создание колонки со значениями
df["Hour"].value_counts()

df["Month"] = df["timeStamp"].apply(lambda t:t.month)
df["Month"].value_counts()

df["DayofWeek"] = df["timeStamp"].apply(lambda t:t.dayofweek)
df["DayofWeek"].value_counts()

df["DayofWeek"].unique()
dic = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thutsday',
      4:'Friday', 5:'Saturday', 6:'Sunday'}
df["DayofWeek"] = df["DayofWeek"].map(dic) # Замена цифр на текстовое значение
df["DayofWeek"].unique()
df.head().T
df["Date"] = df["timeStamp"].apply(lambda t: t.date())
df["twp"].unique()
df["twp"].value_counts().head()
c = (df[df["Reason"] == "EMS"]).groupby("Date").count()
# c["twp"].plot()
dmap = {'NEW HANOVER': "SHYMKENT", "LOWER MERION" : "ALMATY"}

df["City"] = df["twp"].map(dmap)
df["City"].unique()

d = df.groupby("Month").count()
d.head()