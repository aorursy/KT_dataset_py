import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/vgsales.csv")

df.head()
df.info()
df2 = df.groupby('Platform')



df2.head()
plt.figure(figsize=(20,10))

plt.title("Game Global Sales by Platform", fontsize=20)

plt.xlabel("Year of the game's release", fontsize=16)

plt.ylabel("Million Dollers", fontsize=16)

l_keys = []

for key, group in df2:

    l_keys.append(key)

    gy = group.groupby('Year').sum()

    yv = gy[gy.Global_Sales == gy.Global_Sales.max()].Global_Sales.values[0]

    xv = list(gy[gy.Global_Sales == gy.Global_Sales.max()].index)[0]

    x = list(gy.index)

    y = list(gy.Global_Sales.values)

    plt.plot(x, y)

    plt.text(xv, yv, key, fontsize=14)

plt.legend(l_keys)    

# PS series's sales / total sales

(df2.sum().iloc[15:21, :].sum().Global_Sales)/(df2.sum().sum().Global_Sales)
df3 = df.groupby('Year').sum()

df3.plot(kind="area", y=["Other_Sales", "JP_Sales", "EU_Sales", "NA_Sales"], figsize=(20, 10))

plt.title("Market size trend by area", fontsize=20)

plt.xlabel("Year of the game's release", fontsize=16)

plt.ylabel("Million Dollers", fontsize=16)
df4 = df.groupby('Publisher').sum()

df4 = df4[df4['Global_Sales'] >= 100].sort_values('Global_Sales', ascending=False)

df4.plot(kind="bar", y=["Other_Sales", "JP_Sales", "EU_Sales", "NA_Sales"], figsize=(20, 10))

plt.title("Sales by publisher", fontsize=20)

plt.xlabel("Publisher", fontsize=16)

plt.ylabel("Million Dollers", fontsize=16)

print(df4.index)
df.sort_values(by="Global_Sales", ascending=False).head(10)
df.sort_values(by="JP_Sales", ascending=False).head(10)
df.sort_values(by="Other_Sales", ascending=False).head(10)
df.sort_values(by="EU_Sales", ascending=False).head(10)
df.sort_values(by="NA_Sales", ascending=False).head(10)
df5 = df.groupby('Genre').sum()

df5.plot(kind="bar", y=["Other_Sales", "JP_Sales", "EU_Sales", "NA_Sales"], figsize=(20, 10))

plt.title("Sales by genre", fontsize=20)

plt.xlabel("genre", fontsize=16)

plt.ylabel("Million Dollers", fontsize=16)
plt.figure(figsize=(20,10))

plt.title("Game Global Sales by Genre", fontsize=20)

plt.xlabel("Year of the game's release", fontsize=16)

plt.ylabel("Million Dollers", fontsize=16)

l_keys = []

for key, group in df.groupby('Genre'):

    l_keys.append(key)

    gy = group.groupby('Year').sum()

    yv = gy[gy.Global_Sales == gy.Global_Sales.max()].Global_Sales.values[0]

    xv = list(gy[gy.Global_Sales == gy.Global_Sales.max()].index)[0]

    x = list(gy.index)

    y = list(gy.Global_Sales.values)

    plt.plot(x, y)

    plt.text(xv, yv, key, fontsize=14)

plt.legend(l_keys)    
plt.figure(figsize=(20,30))

l_keys = []

for c, market in enumerate(["Other_Sales", "JP_Sales", "EU_Sales", "NA_Sales"]):

    for key, group in df.groupby('Genre'):

        l_keys.append(key)

        gy = group.groupby('Year').sum()

        yv = gy[gy[market] == gy[market].max()][market].values[0]

        xv = list(gy[gy[market] == gy[market].max()].index)[0]

        x = list(gy.index)

        y = list(gy[market].values)

        plt.subplot(4,1,c+1)

        plt.plot(x, y)

        plt.text(xv, yv, key, fontsize=14)

        

    plt.title("Game Global Sales by Genre by %s"%market, fontsize=20)

    plt.xlabel("Year of the game's release", fontsize=16)

    plt.ylabel("Million Dollers", fontsize=16)

    plt.legend(l_keys)    