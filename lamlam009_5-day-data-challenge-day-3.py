# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind

from scipy.stats import probplot

import matplotlib.pyplot as plt

import pylab



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
museums = pd.read_csv("../input/museums.csv").dropna(subset=["Revenue"])

museums = museums[(museums.Revenue != 0)]
museums.head(5)
probplot(museums["Revenue"], dist="norm", plot=pylab)
museums["Museum Type"].unique()
zoos = museums["Revenue"][museums["Museum Type"] == "ZOO, AQUARIUM, OR WILDLIFE CONSERVATION"]

other = museums["Revenue"][museums["Museum Type"] != "ZOO, AQUARIUM, OR WILDLIFE CONSERVATION"]
ttest_ind(zoos, other, equal_var=False)
# let's look at the means (averages) of each group to see which is larger

print("Mean revenue for zoos:")

print(zoos.mean())



print("Mean revenue for others:")

print(other.mean())
# plot the cold cereals

plt.hist(other, alpha=0.5, label='other')

# and the hot cereals

plt.hist(zoos, label='zoos')

# and add a legend

plt.legend(loc='upper right')

# add a title

plt.title("Revenus")
other.describe()
zoos.describe()