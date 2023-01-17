# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/GlobalLandTemperaturesByMajorCity.csv")

df.head()
df.describe()
plt.scatter(df["City"].index, df["AverageTemperature"])
df_sub = df[["City", "AverageTemperature"]]

df_sub.head()
group = df_sub.groupby(by=df_sub["City"], as_index=False).mean()

group.sort_values("AverageTemperature", ascending=False).head(15)
plt.scatter(group["City"].index, group["AverageTemperature"])
plt.boxplot(group["AverageTemperature"])
df_dt_City = df[["dt", "City", "AverageTemperature"]]

df_dt_City.dtypes

wuhan = df_dt_City[df_dt_City["City"].isin(["Wuhan"])]

wuhan.head()

group.head()

group["AverageTemperature"].shape
wuhan["AverageTemperature"].shape

plt.scatter(pd.DatetimeIndex(wuhan["dt"]), wuhan["AverageTemperature"])