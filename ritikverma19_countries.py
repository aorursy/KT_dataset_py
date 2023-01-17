# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/countries-of-the-world/countries of the world.csv")
data.shape
data.head(20)
data.columns
data.info()
data["Region"] = data["Region"].astype("category")
data["Area (sq. mi.)"] = data["Area (sq. mi.)"].astype("float")
def convertNum(listold):
    newlist = []
    for i in listold:
        if(str(i).lower != "nan"):
            newlist.append(float(str(i).replace(',', '.')))
        else:
            newlist.append(np.nan)
    return newlist
data["Coastline (coast/area ratio)"] = convertNum(data["Coastline (coast/area ratio)"])
data["Pop. Density (per sq. mi.)"] = convertNum(data["Pop. Density (per sq. mi.)"])
data["Infant mortality (per 1000 births)"] = convertNum(data["Infant mortality (per 1000 births)"])
data["Net migration"] = convertNum(data["Net migration"])
data["Literacy (%)"] = convertNum(data["Literacy (%)"])
data["Arable (%)"] = convertNum(data["Arable (%)"])
data["Crops (%)"] = convertNum(data["Crops (%)"])
data["Other (%)"] = convertNum(data["Other (%)"])
data["Birthrate"] = convertNum(data["Birthrate"])
data["Deathrate"] = convertNum(data["Deathrate"])
data["Agriculture"] = convertNum(data["Agriculture"])
data["Industry"] = convertNum(data["Industry"])
data["Service"] = convertNum(data["Service"])
data["Phones (per 1000)"] = convertNum(data["Phones (per 1000)"])
data["Climate"] = convertNum(data["Climate"])
data["Climate"] = data["Climate"].astype("category")
data.info()
data.head(10)
data.info()
# Every column after netmigration contains missing values, filling them up with mean values for numerical data and most occurence for categorical data.
data["Net migration"].fillna(data["Net migration"].mean(), inplace=True)
data["Infant mortality (per 1000 births)"].fillna(data["Infant mortality (per 1000 births)"].mean(), inplace=True)
data["GDP ($ per capita)"].fillna(data["GDP ($ per capita)"].mean(), inplace=True)
data["Literacy (%)"].fillna(data["Literacy (%)"].mean(), inplace=True)
data["Phones (per 1000)"].fillna(data["Phones (per 1000)"].mean(), inplace=True)
data["Arable (%)"].fillna(data["Arable (%)"].mean(), inplace=True)
data["Crops (%)"].fillna(data["Crops (%)"].mean(), inplace=True)
data["Other (%)"].fillna(data["Other (%)"].mean(), inplace=True)
data["Birthrate"].fillna(data["Birthrate"].mean(), inplace=True)
data["Deathrate"].fillna(data["Deathrate"].mean(), inplace=True)
data["Agriculture"].fillna(data["Agriculture"].mean(), inplace=True)
data["Industry"].fillna(data["Industry"].mean(), inplace=True)
data["Service"].fillna(data["Service"].mean(), inplace=True)
data["Climate"].value_counts()
data["Climate"].fillna(2.0, inplace=True)
data.info()
data.describe()
import matplotlib.pyplot as plt
import seaborn as sns
total = data.shape[0]
columnsToBeAnalyzed = data.columns[2:]
for i in columnsToBeAnalyzed:
    print(i)
    if(data[i].dtype.name == "category"):
        print((data[i].value_counts()/total)*100)
        sns.countplot(data[i])
    else:
        try:
            sns.distplot(data[i])
        except:
            sns.distplot(data[i], kde=False)
        print("Skew =", data[i].skew())
        print("Kurtosis =", data[i].kurt())
    plt.show()
for i in columnsToBeAnalyzed:
    if(data[i].dtype.name == "category"):
        print(i)
        sns.countplot(data['GDP ($ per capita)'], hue=data[i])

        pd.crosstab(data[i], data['GDP ($ per capita)']).apply(lambda r: round((r/r.sum())*100,1), axis=1)
    plt.show()
for i in columnsToBeAnalyzed:
    if(data[i].dtype.name != "category"):
        print(i)
        sns.relplot(x=i, y="GDP ($ per capita)", data=data)
    plt.show()
sns.heatmap(data.corr())
