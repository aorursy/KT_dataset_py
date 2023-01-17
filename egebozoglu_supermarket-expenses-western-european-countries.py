import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
import pandas as pd

data = pd.read_csv("../input/data.csv")

data
data.drop("Rank",axis=1, inplace=True)
data.columns[0]
data.drop(data.columns[0], axis=1,inplace=True)
data.columns.values
data["Total"] = data.sum(axis=1)

data
data.City.unique()
def Germany(country):

    if "germany" in country.lower():

        return True

    return False



dataGermany = data[data["City"].apply(Germany)]
dataGermany
GerMean = round(dataGermany["Total"].mean())

GerMean
def Austria(country):

    if "austria" in country.lower():

        return True

    return False

dataAustria = data[data["City"].apply(Austria)]
dataAustria
AustMean = round(dataAustria["Total"].mean())

AustMean
def Switzerland(country):

    if "switzerland" in country.lower():

        return True

    return False



dataSwitzerland = data[data["City"].apply(Switzerland)]
dataSwitzerland
SwitzMean = round(dataSwitzerland["Total"].mean())

SwitzMean
def France(country):

    if "france" in country.lower():

        return True

    return False



dataFrance = data[data["City"].apply(France)]
dataFrance
FranceMean = round(dataFrance["Total"].mean())

FranceMean
def Belgium(country):

    if "belgium" in country.lower():

        return True

    return False



dataBelgium = data[data["City"].apply(Belgium)]
dataBelgium
BelgMean = round(dataBelgium["Total"].mean())

BelgMean
def Netherlands(country):

    if "netherlands" in country.lower():

        return True

    return False



dataNeth = data[data["City"].apply(Netherlands)]
dataNeth
NethMean = round(dataNeth["Total"].mean())

NethMean
x = ["Germany", "Austria", "Switzerland", "France", "Belgium", "Netherlands"]

y = [GerMean, AustMean, SwitzMean, FranceMean, BelgMean, NethMean]
plt.figure(figsize=(18,6))

plt.bar(x,y)

plt.ylim(50,160)

plt.xlabel("Countries",size=18)

plt.ylabel("Average Expenses (Euro)", size=18)

plt.title("Supermarket Expenses in Western Europe",size=25)

plt.show()