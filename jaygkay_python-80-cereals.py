# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
cereal = pd.read_csv("../input/cereal.csv")

cereal.head(20)
# summarize our dataset

cereal.describe()
import matplotlib.pylab as plt

# list columns

print(cereal.columns)

sodium = cereal["sodium"]

#plot a histogram

plt.hist(sodium)

plt.title("sodium in 80 cereals products")
# another way of plotting histogram

cereal.hist(column = "sodium", figsize = (12,12))
# plot a histogram of cereal sodium with 9 bins, a black edge around the columns and at a larger size

plt.hist(sodium, bins=9, edgecolor = "black")

plt.title("Sodium in 80 cereals products")

# label

plt.xlabel("sodium in 80 cereals")

plt.ylabel("count")
# import t-test, qqplot

from scipy.stats import ttest_ind, probplot 

import matplotlib.pyplot as plt

import pylab
# plot a qqplot to check nomality

probplot(cereal["sodium"], dist = "norm", plot = pylab)
# get the hot cereals

hotcereals = cereal["sodium"][cereal["type"] == "H"]

# get the cold cereals

coldcereals = cereal["sodium"][cereal["type"] == "C"]



# compare the two variables

ttest_ind(hotcereals, coldcereals, equal_var = False)
# check the means of cereals

print("Mean sodium for the hot cereals")

print("The mean for hot-cereals is", hotcereals.mean())

print("The mean for cold-cereals is", coldcereals.mean())
# plot hot cereals

plt.hist(hotcereals, alpha = 0.5, label = 'hot')

# plot cold cereals

plt.hist(coldcereals, label = 'cold')

# add legend

plt.legend(loc = "upper right")

# add a title

plt.title("sodiumn of 80 cereals by type")
# plot cold cereals

plt.hist(coldcereals, alpha = 0.5, label = 'cold')

# plot hot cereals

plt.hist(hotcereals, label = 'hot')

# add legend

plt.legend(loc = "upper right")

# add a title

plt.title("sodiumn of 80 cereals by type")