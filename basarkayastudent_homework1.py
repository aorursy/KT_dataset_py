# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")
data
data.corr()
# correlation map



f, ax = plt.subplots(figsize = (10,10))

sns.heatmap(data.corr(), annot = True, linewidth=.5, fmt=".1f",ax=ax)

plt.show()
#line plot



data.head(2500).NA_Sales.plot(kind = "line", color="blue", label = "NA_Sales", linewidth=1, alpha=0.7, grid = True, linestyle="-",figsize=(19,19))

data.head(2500).JP_Sales.plot(kind="line", color="red", label = "JP_Sales", linewidth=1, alpha=0.7, grid = True, linestyle=":",figsize=(19,19))

plt.legend(loc="upper left")

plt.xlabel("Games")

plt.ylabel("Sales")

plt.show()
# scatter plot for correlation



data.plot(kind = "scatter", color = "green", x = "EU_Sales", y= "NA_Sales", alpha=0.3, figsize=(13,13))

plt.xlabel("Europe Sales in Millions")

plt.ylabel("North America Sales in Millions")

plt.title("EU-NA")

plt.show()

#histogram for distribution



data.Global_Sales.plot(kind="hist", bins=50 , figsize=(15,15))

plt.xlabel("Sales Number in Millions")

plt.show()
GS = data["Global_Sales"]

NS = data["NA_Sales"]

JS = data["JP_Sales"]

ES = data["EU_Sales"]

OS = data["Other_Sales"]

#these are example of series

RNG = (sum(NS)/sum(GS))*100

RJG = (sum(JS)/sum(GS))*100

REG = (sum(ES)/sum(GS))*100

ROG = (sum(OS)/sum(GS))*100

print("Ratio of North America's Total Sales to Total Global Sales: %", format(RNG, ".2f"))

print("---------------------------------------------------------------------------------------")

print("Ratio of Europe's Total Sales to Total Global Sales: %", format(REG, ".2f"))

print("---------------------------------------------------------------------------------------")

print("Ratio of Japan's Total Sales to Total Global Sales: %", format(RJG, ".2f"))

print("---------------------------------------------------------------------------------------")

print("Ratio of Other Countries' Total Sales to Total Global Sales: %", format(ROG, ".2f"))
data
x = data["NA_Sales"]>9.0

data[x]
data[np.logical_and(data["NA_Sales"]> 8.0, data["JP_Sales"]> 3.0)]

print(data["Genre"].value_counts())
print(data.Publisher.value_counts())
data.describe()
data1 = data.head()

data1
melted = pd.melt(frame = data1, id_vars="Name", value_vars = "Global_Sales")

melted
pd.melt(frame = data, id_vars="Name", value_vars = "Global_Sales")
melted.pivot(index = "Name", columns="variable", values= "value")
data1 = data.head()

data2 = data.tail()

conc_data = pd.concat([data1,data2], axis=0, ignore_index=True)

conc_data
data3 = pd.concat([data.head().Name, data1.Global_Sales,data1.Genre],axis=1)

data3
data.dtypes
data["Platform"] = data["Platform"].astype("category")
data.info()
data.Year.value_counts(dropna=False)
data1 = data

data1.Year.dropna(inplace=True)

assert data.Year.notnull().all()
data.Year.fillna("empty",inplace=True)
assert data.Year.notnull().all()
data.Year.value_counts(dropna=False)
dataplot = data.head(250).loc[:,["Global_Sales","NA_Sales","JP_Sales"]]

dataplot.plot()

plt.show()
dataplot.plot(subplots=True)

plt.show()
data.head(500).plot(kind="scatter",x = "NA_Sales",y = "EU_Sales")

plt.show()
fig,axes = plt.subplots(nrows=2, ncols=1)

dataplot.plot(kind="hist", y = "JP_Sales", bins=50, normed=True, ax=axes[0])

dataplot.plot(kind="hist", y = "JP_Sales", bins=50, normed=True, ax=axes[1],cumulative=True)

plt.show()
data = data.set_index("Rank")

data
data.Platform[1]
data.loc[:,["Publisher"]]
data[["Name","Other_Sales"]]
data.loc[:15,["NA_Sales","EU_Sales"]]
boolean = data.Global_Sales > 20

data[boolean]
filter1 = data.NA_Sales > 5

filter2 = data.JP_Sales < 1

data[filter1 & filter2]
data.Year.value_counts()
def div(x):

    return x*2



data.JP_Sales.apply(div)
data.JP_Sales.apply(lambda x: x*2)
data
del data["NA_EU_Sales"]
data
data.index.name
data.index.name = "index"
data
data.index.name = "Rank"
data1 = data.set_index(["Publisher","Name"])

data1.head(40)