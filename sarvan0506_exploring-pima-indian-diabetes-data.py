# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# For Plotting

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv("../input/diabetes.csv")

data.head()
data.describe()
percent = (data.groupby("Outcome")["Outcome"].count()/len(data.index))*100

plt.figure(figsize=(5,5))

plt.pie(percent, labels=(percent.round(2)))
sns.pairplot(data, hue="Outcome")
sns.heatmap(data.drop("Outcome", axis=1).corr(), annot=True)
plt.hist(data["Age"])
bins = pd.Series([])

for i in data.index:

    if (data.loc[i:i,]["Age"] <= 25).bool(): bins = bins.append(pd.Series(["20-25"]))

    elif (data.loc[i:i,]["Age"] <= 30).bool(): bins = bins.append(pd.Series(["26-30"]))

    elif (data.loc[i:i,]["Age"] <= 35).bool(): bins = bins.append(pd.Series(["31-35"]))

    elif (data.loc[i:i,]["Age"] <= 40).bool(): bins = bins.append(pd.Series(["36-40"]))

    elif (data.loc[i:i,]["Age"] <= 45).bool(): bins = bins.append(pd.Series(["41-45"]))

    elif (data.loc[i:i,]["Age"] <= 50).bool(): bins = bins.append(pd.Series(["46-50"]))

    elif (data.loc[i:i,]["Age"] <= 55).bool(): bins = bins.append(pd.Series(["51-55"]))

    elif (data.loc[i:i,]["Age"] <= 60).bool(): bins = bins.append(pd.Series(["56-60"]))

    elif (data.loc[i:i,]["Age"] <= 65).bool(): bins = bins.append(pd.Series(["61-65"]))

    else: bins = bins.append(pd.Series([">65"]))

bins.reset_index(drop=True, inplace=True)

data["Ages"] = bins

data.head()
bindata1 = data[data["Outcome"]==1].groupby("Ages")[["Outcome"]].count()

bindata1.head()
bindata = data.groupby("Ages")[["Outcome"]].count()

bindata1["% Diabetic"] = (bindata1["Outcome"]/bindata["Outcome"])*100
sns.barplot(x=bindata1.index, y=bindata1["% Diabetic"])
fig = plt.figure(figsize=(20,3))

for i in np.arange(1,7):

        splt =  plt.subplot(1,7,i,title=data.columns[i])

        plt.boxplot(data[data.columns[i]])
gluData = data[data["Glucose"]!=0]
bins = np.arange(min(gluData["Glucose"]),max(gluData["Glucose"]),10)

bins
gluData["Glucose Levels"] = pd.cut(gluData["Glucose"], bins=bins)

gluData.head()
bindata1 = gluData[gluData["Outcome"]==1].groupby("Glucose Levels")[["Outcome"]].count()

bindata = gluData.groupby("Glucose Levels")[["Outcome"]].count()

bindata1["% Diabetic"] = (bindata1["Outcome"]/bindata["Outcome"])*100
plt.figure(figsize=(15,5))

sns.barplot(x=bindata1.index, y=bindata1["% Diabetic"])
pressData = data[data["BloodPressure"]!=0]
bins = np.arange(min(pressData["BloodPressure"]),max(pressData["BloodPressure"]),10)

pressData["BP Levels"] = pd.cut(pressData["BloodPressure"], bins=bins)

bindata1 = pressData[pressData["Outcome"]==1].groupby("BP Levels")[["Outcome"]].count()

bindata = pressData.groupby("BP Levels")[["Outcome"]].count()

bindata1["% Diabetic"] = (bindata1["Outcome"]/bindata["Outcome"])*100

plt.figure(figsize=(15,5))

sns.barplot(x=bindata1.index, y=bindata1["% Diabetic"])