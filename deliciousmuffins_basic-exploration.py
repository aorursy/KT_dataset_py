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
df = pd.read_csv("../input/banks.csv")

df.head()

df.count()
df = df.join( df["Headquarters"].str.split(",", expand = True))

df["D/A"] = df["Total Deposits"]/df["Total Assets"]

df["Failure Date"] = pd.to_datetime(df["Failure Date"])

df = df.rename(columns = {0:"City", 1:"State", 2:"Del"})

df = df.drop("Del", 1)

df.count()
fails_by_year = df['Failure Date'].groupby([df["Failure Date"].dt.year]).agg('count')

plt.figure(figsize=(15,10))

sns.barplot(fails_by_year.index, fails_by_year)

plt.xticks(rotation="vertical")

plt.show()
plt.figure(figsize=(8,6))

df.groupby("State").count()["Failure Date"].sort_values(ascending=False)[0:25].plot(kind="bar")

plt.ylabel("Total Failures")

plt.show()
df.groupby("State").sum()["Estimated Loss (2015)"].sort_values(ascending=False)[0:25].plot(kind="bar")

plt.ylabel("Total Estimated Losses")

plt.show()
df[df["State"] == " TX"].sort_values(by = "Estimated Loss (2015)", ascending = False)
plt.figure(figsize=(8,6))

df.groupby("City").count()["Failure Date"].sort_values(ascending=False)[0:25].plot(kind="bar")

plt.ylabel("Total Failures")

plt.show()
plt.figure(figsize=(8,6))

df.groupby("State").mean()["D/A"].sort_values(ascending=False)[0:50].plot(kind="bar")

plt.ylabel("Mean Deposit to Asset Ratio")

plt.show()
plt.figure(figsize=(8,6))

plt.scatter(x = df["D/A"], y = df["Estimated Loss (2015)"])

plt.xlabel("Deposits to Asset Ratio")

plt.ylabel("Estimated Loss")

plt.show()
plt.figure(figsize=(8,6))

plt.plot_date(x = df["Failure Date"], y = df["D/A"])

plt.ylabel("Deposits to Asset Ratio")

plt.show()