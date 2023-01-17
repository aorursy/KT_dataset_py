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
# Importing required packages

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

# Reading the export dataset into export variable

export = pd.read_csv("../input/india-trade-data/2018-2010_export.csv")

export.head()
# Finding the shape of our dataset

export.shape

rows_export,column_export = export.shape

print(rows_export)

print(column_export)
# Checking the non-null values in dataframe

export.info()

# We can see that thevalue column has only 122985 non-null values so it has around 14000 null-values
# To check the IQR,min,max,mean,standard-dev

export.describe()
# To find the maximum value that value column has

export["value"].max()
# To find the minimum value that value column has

export["value"].min()
# To retrieve all the rows that have value as 0

export.loc[export["value"]==0.0].reset_index()
# To retrieve the row that has maximum value

export.loc[export["value"]==export["value"].max()].reset_index()
# To retrieve the total values in HSCode

export["HSCode"].value_counts()
pd.set_option("max_columns",100,"max_rows",300)

# To retrieve the number of country where products are exported

export["country"].value_counts()
s = export.groupby("country")[["value"]]

s
for country,country_value in s:

    print(country)

    print(country_value)
a = s.sum()

a
a.loc[a["value"]==a["value"].max()]
a.loc[a["value"]==a["value"].min()]
a.loc[a["value"]>=a["value"].mean()]
a.loc[a["value"]<=a["value"].median()]
a.loc[a["value"]>=a["value"].median()]
export.head()
pd.set_option("max_columns",100,"max_rows",300)

a = export.groupby(["country","HSCode"])[["value"]].sum()

a
export.groupby(["country","HSCode"])["value"].count()
export.groupby(["country","HSCode"]).agg({"value" :sum,"value":"count","value":"first"})
export[["value","country"]]
export.isna().sum()
export[export.isnull().any(axis=1)].reset_index()
# counting the no of times NAN values occured for nam

plt.figure(figsize=(100,20))

a = (export.groupby(["HSCode"])["value"].sum()).plot(kind="bar")

a.set_xticklabels(a.get_xticklabels(), rotation=45, rotation_mode="anchor")
# Filling the NAN values as 0 in value column 

export["value"].fillna(value=0.0,inplace=True)
export.isna().sum()
a = sns.pairplot(data=export,hue="year")

a.add_legend()
# Data Visualization using Seaborn

# Creating a simple countplot

# plt.figure(figsize=(30,8))

# s.sum().plot(kind="bar")



export_by_year = export.groupby("year")

export_by_year
export_by_year.describe()
plt.figure(figsize=(30,10))

sns.heatmap(export_by_year.describe(),annot=True,fmt="n",cmap="twilight")
plt.figure(figsize=(30,10))

sns.clustermap(export.corr(),annot=True,fmt="n",cmap="RdBu")
plt.figure(figsize=(30,10))

sns.heatmap(export.corr(),annot=True,fmt="n",cmap="PuBu")
export_by_year = export.groupby("year")[["value"]].sum()

export_by_year
export_by_year.plot()
export_by_yh = export.groupby(["year","HSCode"])[["value"]].sum()

export_by_yh
plt.figure(figsize=(30,10))

a = export_by_yh.plot(kind="line")

a.set_xticklabels(a.get_xticklabels(), rotation=45, rotation_mode="anchor")
t = export.groupby(["year","country"])["value"].sum().plot()

t.set_xticklabels(t.get_xticklabels(),rotation=90)