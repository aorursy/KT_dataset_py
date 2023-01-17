# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



df = pd.read_csv("../input/tips.csv", encoding="ISO-8859-1")



print ("Dataset columns")

print (list(df.columns))

print ("")



print ("Dropping both 'UID' and 'ID'")

df = df.drop(["UID", "ID"], axis=1)

print ("")



print ("Dataset shape")

print (list(df.shape))

print("")



print ("Dataset head")

print (df.head())

print ("")



# If you want to describe every column on the dataset, use include="all" on describe

# It`d describe every column as a numeric values

# print (df.describe(include="all"))



# We are going to separete those columns by type



print ("Describing numeric values")

print (df.describe(include=[np.number]))

print ("")



print ("Describing string columns")

# Date, Bet type and Result are categorical columns

print (df.drop(["Date", "Bet Type", "Result"], axis=1).describe(include=[np.object]))

print ("")



print ("Describing categorical columns")

print (df.drop(["Date", "Tipster", "Track", "Horse"], axis=1).describe(include=[np.object]))

print ("")
sns.countplot(df.TipsterActive)

plt.figure()

sns.countplot(df['Bet Type'])