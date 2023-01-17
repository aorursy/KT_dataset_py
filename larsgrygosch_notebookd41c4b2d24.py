# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Wiederholung

x = np.arange(20)
y1 = 2*x + np.random.normal(10, 2, 20)
y2 = 3*x + np.random.normal(10, 2, 20)

data = np.array([x, y1, y2]).T
data
df = pd.DataFrame(data)
df.columns = ["x", "y1", "y2"]
df.tail()
df.info()
df.describe()
df.corr()
df["y1"]
df[["y1", "y2"]]
df.iloc[2, 2]
df.loc[2, "y2"]
df[df["y2"] > 20][df["y2"] < 30]
df.hist()
df["div"] = df["y1"] - df["y2"]
df.head()
import matplotlib.pyplot as plt
plt.plot(x, y1, color="r")
plt.plot(x, y2, color="k")

plt.title("einfacher Plot")
plt.xlabel("x-Achse")
plt.ylabel("y-Achse")

plt.show()
plt.scatter(x, y1)
plt.scatter(x,y2)
plt.plot(np.arange(20), np.arange(20)*2 + 10)
plt.show()
from random import randint, choice

dates = [int(choice(["198402", "202002"]) + str(randint(10,20))) for i in range(50)]
vals = np.random.normal(12, 2, 50)

df = pd.DataFrame({"dates": dates, "vals": vals})
df.head()
df.info()
type(df.iloc[0,0])
test_str = "19840210"

print(test_str.startswith("2020"))

test_in = int(test_str)
print(type(test_in))
df["dates"] = df["dates"].astype(str)
df.info()
df_filter = df
df_filter = df_filter[df_filter["dates"].str.startswith("2020")]
df_filter.info()
df.head()
test_str.replace("198402", "")
df_filter["dates"] = df_filter.loc[: , "dates"].str.replace("202002", "")
df_filter
plt.plot(df_filter["dates"], df_filter["vals"])
plt.show()
df_sorted = df_filter.sort_values(by=["dates"])
df_sorted.head()
plt.plot(df_sorted["dates"], df_sorted["vals"])
plt.show()
df_sorted.groupby("dates").mean()
plt.plot(df_sorted.groupby("dates").mean())
#1 Datensatz öffnen pd.read_csv("data.csv", sep=";") 
#2 Untersuchen
#3 Datum, Temperatur (TT_TU), Luftfeuchtig (RF_TU)
#4 Wetterdaten an deinem Geburtstag
#5 Finde die durchschnittstemperatur zw. 6-18 Uhr
#6 plot ür Temp und Luft