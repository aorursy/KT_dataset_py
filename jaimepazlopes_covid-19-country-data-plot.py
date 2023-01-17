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
country1 = "Brazil"

country2 = "Italy"
data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

data.head()
data = data[(data["Country_Region"] == country1) & (data["ConfirmedCases"] > 0)]

data.head()
data.info()
data.drop(["Province_State", "Id", "Country_Region"], axis=1, inplace=True)

data.info()
data.reset_index(inplace = True, drop=True)

data["Day"] = data.index + 1
data.head(39)
data["IncresedCases"] = 0

for index in range(1, len(data)):

    data.loc[index, "IncresedCases"] = data.loc[index, "ConfirmedCases"] / data.loc[index-1, "ConfirmedCases"] - 1
data["IncresedFatalities"] = 0

for index in range(1, len(data)):

    if data.loc[index, "Fatalities"] == 0 or data.loc[index-1, "Fatalities"] == 0:

        data.loc[index, "IncresedFatalities"] = 0

    else:

        data.loc[index, "IncresedFatalities"] = data.loc[index, "Fatalities"] / data.loc[index-1, "Fatalities"] - 1

data.head(1)
data["NewCases"] = 0

for index in range(1, len(data)):

    data.loc[index, "NewCases"] = data.loc[index, "ConfirmedCases"] - data.loc[index-1, "ConfirmedCases"]



data["NewFatalities"] = 0

for index in range(1, len(data)):

    data.loc[index, "NewFatalities"] = data.loc[index, "Fatalities"] - data.loc[index-1, "Fatalities"]
import matplotlib.pyplot as plt

import seaborn as sns

f, ax = plt.subplots(nrows=1, ncols=2, figsize=(24,6))

ax[0].plot(data["Day"],data["ConfirmedCases"], "b.-", label="ConfirmedCases", markersize=15)

ax[0].plot(data["Day"],data["Fatalities"], "r.-", label="Fatalities", markersize=15)

ax[0].set_xlabel('Day')

ax[0].set_ylabel('People')

ax[0].set_title(f'{country1}: Confirmed Cases X Fatalities - Linear')

ax[0].legend(ncol=2, loc="upper left", frameon=True)



ax[1].set(yscale="log")

ax[1].plot(data["Day"],data["ConfirmedCases"], "b.-", label="ConfirmedCases", markersize=15)

ax[1].plot(data["Day"],data["Fatalities"], "r.-", label="Fatalities", markersize=15)

ax[1].set_xlabel('Day')

ax[1].set_ylabel('People')

ax[1].set_title(f'{country1}: Confirmed Cases X Fatalities - Logarithmic')

ax[1].legend(ncol=2, loc="upper left", frameon=True)
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(24,6))

sns.barplot(x="Day", y="ConfirmedCases", data=data, color="b", label="Confirmed Cases", ax=ax[0])

sns.barplot(x="Day", y="Fatalities", data=data, color="r", label="Fatalities", ax=ax[0])

ax[0].set_xlabel('Day')

ax[0].set_ylabel('People')

ax[0].set_title(f'{country1}: Confirmed Cases X Fatalities - Linear')

ax[0].legend(ncol=2, loc="upper left", frameon=True)



ax[1].set(yscale="log")

sns.barplot(x="Day", y="ConfirmedCases", data=data, color="b", label="Confirmed Cases")

sns.barplot(x="Day", y="Fatalities", data=data, color="r", label="Fatalities")

ax[1].set_xlabel('Day')

ax[1].set_ylabel('People')

ax[1].set_title(f'{country1}: Confirmed Cases X Fatalities - Logarithmic')

ax[1].legend(ncol=2, loc="upper left", frameon=True)
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(24,6))

sns.barplot(x="Day", y="NewCases", data=data, color="#808080", label="Confirmed Cases", ax=ax[0])

ax[0].set_xlabel('Day')

ax[0].set_ylabel('People')

ax[0].set_title(f'{country1}: New Confirmed Cases')



sns.barplot(x="Day", y="NewFatalities", data=data, color="#808080", label="Fatalities")

ax[1].set_xlabel('Day')

ax[1].set_ylabel('People')

ax[1].set_title(f'{country1}: New Confirmed Fatalities')
fig, ax = plt.subplots(figsize=(12,6))

sns.barplot(x="Day", y="NewCases", data=data, color="b", label="Confirmed Cases")

sns.barplot(x="Day", y="NewFatalities", data=data, color="r", label="Fatalities")

ax.set_xlabel('Day')

ax.set_ylabel('People')

ax.set_title(f'{country1}: New Confirmed Cases X New Fatalities')

ax.legend(ncol=2, loc="upper left", frameon=True)
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(24,6))

ax[0].plot(data["Day"],data["IncresedCases"]*100, "b.-", label="ConfirmedCases")

ax[0].set_xlabel('Day')

ax[0].set_ylabel('People')

ax[0].set_title(f'{country1}: Daily Increased Cases - Percentage')

ax[0].legend(ncol=2, loc="upper left", frameon=True)



ax[1].plot(data["Day"],data["IncresedFatalities"]*100, "r.-", label="Fatalities")

ax[1].set_xlabel('Day')

ax[1].set_ylabel('People')

ax[1].set_title(f'{country1}: Daily Increased Fatalities - Percentage')

ax[1].legend(ncol=2, loc="upper left", frameon=True)
data2 = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

data2 = data2[(data2["Country_Region"] == country2) & (data2["ConfirmedCases"] > 0)]

len(data2)

data2.head(65)
data2.reset_index(inplace = True, drop=True)

data2["Day"] = data2.index + 1

data2["IncresedCases"] = 0

for index in range(1, len(data2)):

    data2.loc[index, "IncresedCases"] = data2.loc[index, "ConfirmedCases"] / data2.loc[index-1, "ConfirmedCases"] - 1



data2["IncresedFatalities"] = 0 

for index in range(1, len(data2)):     

    if data2.loc[index, "Fatalities"] == 0 or data2.loc[index-1, "Fatalities"] == 0:         

        data2.loc[index, "IncresedFatalities"] = 0     

    else:         

        data2.loc[index, "IncresedFatalities"] = data2.loc[index, "Fatalities"] / data2.loc[index-1, "Fatalities"] - 1



data2["NewCases"] = 0

for index in range(1, len(data2)):

    data2.loc[index, "NewCases"] = data2.loc[index, "ConfirmedCases"] - data2.loc[index-1, "ConfirmedCases"]



data2["NewFatalities"] = 0

for index in range(1, len(data2)):

    data2.loc[index, "NewFatalities"] = data2.loc[index, "Fatalities"] - data2.loc[index-1, "Fatalities"]

    

data2.head(10)



data2.drop(["Province_State", "Id", "Country_Region"], axis=1, inplace=True)

data2.tail(10)

f, ax = plt.subplots(nrows=1, ncols=2, figsize=(24,6))

ax[0].plot(data2["Day"],data2["ConfirmedCases"], "b.-", label="ConfirmedCases", markersize=15)

ax[0].plot(data2["Day"],data2["Fatalities"], "r.-", label="Fatalities", markersize=15)

ax[0].set_xlabel('Day')

ax[0].set_ylabel('People')

ax[0].set_title(f'{country2}: Confirmed Cases X Fatalities - Linear')

ax[0].legend(ncol=2, loc="upper left", frameon=True)



ax[1].set(yscale="log")

ax[1].plot(data2["Day"],data2["ConfirmedCases"], "b.-", label="ConfirmedCases", markersize=15)

ax[1].plot(data2["Day"],data2["Fatalities"], "r.-", label="Fatalities", markersize=15)

ax[1].set_xlabel('Day')

ax[1].set_ylabel('People')

ax[1].set_title(f'{country2}: Confirmed Cases X Fatalities - Logarithmic')

ax[1].legend(ncol=2, loc="upper left", frameon=True)
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(30,6))

sns.barplot(x="Day", y="NewCases", data=data2, color="#808080", label="Confirmed Cases", ax=ax[0])

ax[0].set_xlabel('Day')

ax[0].set_ylabel('People')

ax[0].set_title(f'{country2}: New Confirmed Cases')



sns.barplot(x="Day", y="NewFatalities", data=data2, color="#808080", label="Fatalities")

ax[1].set_xlabel('Day')

ax[1].set_ylabel('People')

ax[1].set_title(f'{country2}: New Confirmed Fatalities')
fig, ax = plt.subplots(figsize=(20,6))

sns.barplot(x="Day", y="NewCases", data=data2, color="b", label="Confirmed Cases")

sns.barplot(x="Day", y="NewFatalities", data=data2, color="r", label="Fatalities")

ax.set_xlabel('Day')

ax.set_ylabel('People')

ax.set_title(f'{country2}: New Confirmed Cases X New Fatalities')

ax.legend(ncol=2, loc="upper left", frameon=True)
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(30,6))

ax[0].plot(data2["Day"],data2["IncresedCases"]*100, "b.-", label="ConfirmedCases")

ax[0].set_xlabel('Day')

ax[0].set_ylabel('People')

ax[0].set_title(f'{country2}: Daily Increased Cases - Percentage')

ax[0].legend(ncol=2, loc="upper left", frameon=True)



ax[1].plot(data2["Day"],data2["IncresedFatalities"]*100, "r.-", label="Fatalities")

ax[1].set_xlabel('Day')

ax[1].set_ylabel('People')

ax[1].set_title(f'{country2}: Daily Increased Fatalities - Percentage')

ax[1].legend(ncol=2, loc="upper left", frameon=True)
f, ax = plt.subplots(figsize=(20,10))

ax.plot(data["Day"],data["ConfirmedCases"], "b*-", label=country1, markersize=15)

ax.plot(data2["Day"],data2["ConfirmedCases"], "g.-", label=country2, markersize=15)

ax.set_xlabel('Day')

ax.set_ylabel('People')

ax.set_title(f'{country1} X {country2}: Confirmed Cases - Linear')

ax.legend(ncol=2, loc="upper left", frameon=True)
smallestsize = min(len(data), len(data2))

smallestsize
f, ax = plt.subplots(figsize=(16,6))

ax.plot(data["Day"][:smallestsize],data["ConfirmedCases"][:smallestsize], "b*-", label=country1, markersize=15)

ax.plot(data2["Day"][:smallestsize],data2["ConfirmedCases"][:smallestsize], "g.-", label=country2, markersize=15)

ax.set_xlabel('Day')

ax.set_ylabel('People')

ax.set_title(f'{country1} X {country2}: Confirmed Cases - Linear')

ax.legend(ncol=2, loc="upper left", frameon=True)
f, ax = plt.subplots(figsize=(12,6))

ax.set(yscale="log")

ax.plot(data["Day"],data["ConfirmedCases"], "b*-", label=country1, markersize=15)

ax.plot(data2["Day"],data2["ConfirmedCases"], "g.-", label=country2, markersize=15)

ax.set_xlabel('Day')

ax.set_ylabel('People')

ax.set_title(f'{country1} X {country2}: Confirmed Cases - Logarithmic')

ax.legend(ncol=2, loc="upper left", frameon=True)
f, ax = plt.subplots(figsize=(20,10))

ax.plot(data["Day"],data["Fatalities"], "b*-", label=country1, markersize=15)

ax.plot(data2["Day"],data2["Fatalities"], "g.-", label=country2, markersize=15)

ax.set_xlabel('Day')

ax.set_ylabel('People')

ax.set_title(f'{country1} X {country2}: Fatalities - Linear')

ax.legend(ncol=2, loc="upper left", frameon=True)
f, ax = plt.subplots(figsize=(16,6))

ax.plot(data["Day"][:smallestsize],data["Fatalities"][:smallestsize], "b*-", label=country1, markersize=15)

ax.plot(data2["Day"][:smallestsize],data2["Fatalities"][:smallestsize], "g.-", label=country2, markersize=15)

ax.set_xlabel('Day')

ax.set_ylabel('People')

ax.set_title(f'{country1} X {country2}: Fatalities - Linear')

ax.legend(ncol=2, loc="upper left", frameon=True)
f, ax = plt.subplots(figsize=(20,10))

ax.set(yscale="log")

ax.plot(data["Day"],data["Fatalities"], "b*-", label=country1, markersize=15)

ax.plot(data2["Day"],data2["Fatalities"], "g.-", label=country2, markersize=15)

ax.set_xlabel('Day')

ax.set_ylabel('People')

ax.set_title(f'{country1} X {country2}: Fatalities - Logarithmic')

ax.legend(ncol=2, loc="upper left", frameon=True)
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(30,6))

ax[0].plot(data["Day"][:smallestsize],data["NewCases"][:smallestsize], "b*-", label=country1, markersize=15)

ax[0].plot(data2["Day"][:smallestsize],data2["NewCases"][:smallestsize], "g.-", label=country2, markersize=15)

ax[0].set_xlabel('Day')

ax[0].set_ylabel('People')

ax[0].set_title(f'{country1} X {country2}: New Confirmed Cases')



ax[1].plot(data["Day"][:smallestsize],data["NewFatalities"][:smallestsize], "b*-", label=country1, markersize=15)

ax[1].plot(data2["Day"][:smallestsize],data2["NewFatalities"][:smallestsize], "g.-", label=country2, markersize=15)

ax[1].set_xlabel('Day')

ax[1].set_ylabel('People')

ax[1].set_title(f'{country1} X {country2}: New Confirmed Fatalities')