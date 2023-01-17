# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns #visualization

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/earthquake/earthquake.csv")
data.head(10)

data.info
data.columns
def yeardate(x):

    return x[0:4]

data["yeardate"] = data.date.apply(yeardate)

#We must change object to integer.

data['yeardate'] = data.yeardate.astype(int)

print(data.yeardate.dtypes)

data.head(3)
plt.figure(figsize=(15,15))

sns.heatmap(data.corr(), annot = True, fmt= ".1f", linewidths = .3)

plt.show()
data.yeardate.plot(kind = "hist" , color = "red" , edgecolor="black", bins = 100 , figsize = (12,12) , label = "Earthquakes frequency")

plt.legend(loc = "upper right")

plt.xlabel("Years")

plt.show()
data.city.value_counts().plot(kind = "bar" , color = "red" , figsize = (30,10),fontsize = 20)

plt.xlabel("City",fontsize=18,color="blue")

plt.ylabel("Frequency",fontsize=18,color="blue")

plt.show()
data.country.value_counts().plot(kind = "bar" , color = "red" , figsize = (30,10),fontsize = 20)

plt.xlabel("Country",fontsize=18,color="blue")

plt.ylabel("Frequency",fontsize=18,color="blue")

plt.show()
data.long.max()

filtre = data.long == 48.0

data[filtre]
data.xm.max()

filtering = data.country == "turkey"

filtering2 = data.xm == 7.9

data[filtering & filtering2]
threshold = sum(data.xm) / len(data.xm)

data["magnitude-level"] = ["hight" if i > threshold else "low" for i in data.xm]

data.loc[:10,["magnitude-level","xm","city"]]