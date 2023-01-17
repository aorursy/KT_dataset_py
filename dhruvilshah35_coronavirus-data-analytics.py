# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



dataset = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

print(dataset.head(50))
print(dataset.tail())
print(dataset.info())
print(dataset.describe())
bydate = dataset[["ObservationDate","Confirmed","Deaths","Recovered"]].groupby(dataset["ObservationDate"]).sum()

bydate["Confirmed"][1:] = list(map(lambda i: bydate["Confirmed"][i] - bydate["Confirmed"][i-1],range(1,len(bydate))))

bydate["Deaths"][1:] = list(map(lambda i: bydate["Deaths"][i] - bydate["Deaths"][i-1],range(1,len(bydate))))

bydate["Recovered"][1:] = list(map(lambda i: bydate["Recovered"][i] - bydate["Recovered"][i-1],range(1,len(bydate))))

bydate["ConfirmedDeathRatio"] = list(map(lambda i: f'{bydate["Confirmed"][i] / bydate["Deaths"][i]:.2f}',range(len(bydate))))

print(bydate.head())

print(bydate.tail())

bydate["ConfirmedDeathRatio"] = list(map(float,bydate["ConfirmedDeathRatio"]))

bydate["Confirmed"] = list(map(float,bydate["Confirmed"]))

bydate["Deaths"] = list(map(float,bydate["Deaths"]))

bydate["Recovered"] = list(map(float,bydate["Recovered"]))



print("\nTotal Confirmed Cases: " + str(int(bydate["Confirmed"].sum())))

print("Total Deaths Cases: " + str(int(bydate["Deaths"].sum())))

print("Total Recovered Cases: " + str(int(bydate["Recovered"].sum())))

print("Average of confirmed cases to death ratio: " + str(bydate["ConfirmedDeathRatio"].mean()))
plt.figure(figsize = (15,5))

plt.plot(range(len(bydate)), bydate["Confirmed"])

plt.plot(range(len(bydate)), bydate["Deaths"])

plt.plot(range(len(bydate)), bydate["Recovered"])

plt.xticks(range(50))

plt.legend(loc = 'upper right')

plt.xlabel("Days")

plt.ylabel("Cases")

plt.title("From 01/22/20 to 03/11/20 Cases Overview")

plt.show()
case_type_sum = [int(bydate["Confirmed"].sum()),int(bydate["Deaths"].sum()),int(bydate["Recovered"].sum())]

plt.bar(range(3),case_type_sum, color = ["blue","red","green"])

case_type = ["Confirmed","Deaths","Recovered"]

plt.xticks(range(3),case_type)

plt.ylabel("Cases")

plt.title("From 01/22/20 to 03/11/20 Cases Overview")

plt.show()