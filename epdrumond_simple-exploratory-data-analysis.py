# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Plotting imports

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
"""

This is a basic data exploration aimed at improving my skills. I have taken some tips from other kernels based 

on this data, specially Cristiana Andrada's.

If you have any sugestions on how I can improve this kernel, please tell me. 

"""



# Read data

dbase = pd.read_csv("../input/BRAZIL_CITIES.csv", sep = ";", decimal = ",")

dbase.head()



# Acquire data frame shape

dbase.shape
# Before any data exploration begins, let's look at the missing values

dbase.isnull().sum()



# For the purposes of this analysis, the missing values are going to be replaced according to the data type

for x in range(len(dbase.dtypes)):

    if dbase.dtypes[x] == object: 

        dbase.iloc[:,x].fillna("none", inplace = True)

    else: 

        dbase.iloc[:,x].fillna(0.0, inplace = True)
# First, we extract some basic information



# Cities per State

plt.figure(figsize = (10,6))

sns.countplot(x = "STATE", data = dbase)

# HDI index distribution

sns.distplot(dbase["IDHM"])
# Rural/Urban cities

plt.figure(figsize = (10,6))

plt.xticks(rotation = 90)

sns.countplot(x = "RURAL_URBAN", data = dbase)
# Hotel beds distribution over the country

plt.figure(figsize = (10,10))



mask01 = dbase["LONG"] != 0

mask02 = dbase["LAT"] != 0



x = dbase[mask01 & mask02]["LONG"]

y = dbase[mask01 & mask02]["LAT"]

z = dbase[mask01 & mask02]["BEDS"]



plt.scatter(x, y, s = z/10, alpha = 1)
# Second, we explore the relationship between attributes



# Resident population vs. IDHM

sns.jointplot(x = dbase["IBGE_RES_POP"], y = dbase["IDHM"], kind = "hex")
# Total Number of Companies vs. Total Number of Cars

sns.lineplot(x = "COMP_TOT", y = "Cars", data = dbase)
# GDP per Capita vs. Total number of MacDonalds stores

sns.jointplot(x = dbase["GDP_CAPITA"], y = dbase["MAC"])