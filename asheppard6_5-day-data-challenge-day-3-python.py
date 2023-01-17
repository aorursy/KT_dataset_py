# This kernal shows how to compare two groups by perfoming a t-test



# Load in libraries

#from scipy.stats import ttest_ind as sp # stats, t-test

from scipy import stats

from scipy.stats import ttest_ind

from scipy.stats import probplot

import matplotlib.pyplot as plt # for qqplot

import pylab

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# List files in input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Read data into dataframe

df1 = pd.read_csv("../input/cereal.csv")

# Delete rows with any NaN values

df = df1.dropna(axis = 0, how = 'any')
# check for any NaN values

df.isnull().values.any()

df.isnull().sum()
# Summarize data

df.describe()
# Head data

df.head()
# Check normality (which is an assumption for a t-test)

# Use qqplot. If variable is normally distributed, most of the points will be along the center diagonal.

probplot(df["rating"], dist="norm", plot=pylab)
# t-test: Kellogs vs Nabisco ratings

# get ratings population for Kellogs

Kellogs = df["rating"][df["mfr"] == "K"]

# get ratings population for Nabisco

Nabisco = df["rating"][df["mfr"] == "N"]



stats.ttest_ind(Kellogs, Nabisco, equal_var = False)
# Look at means for each rating group

print("Mean rating for Kellogs products:")

print(Kellogs.mean())



print("Mean rating for Nabisco products:")

print(Nabisco.mean())
# Plot both groups

plt.hist(Kellogs, label = 'Kellogs')

plt.hist(Nabisco, label = 'Nabisco')

plt.legend(loc = 'upper right')

plt.title("Rating of Kellogs vs Nabisco products")