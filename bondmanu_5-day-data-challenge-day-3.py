# importin the libraries and reading the dataset

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind    # for t_test

from scipy.stats import probplot     # for qq plot

import matplotlib.pyplot as plt      # for qq plot

import pylab

import seaborn as sns                # seaborn for histograms

sns.set(color_codes=True)

data = pd.read_csv("../input/cereal.csv")

data.head()



# Any results you write to the current directory are saved as output.
# plot a qq plot to check normality, if the variable is normally distributerd

# most of the points should be along the center diagonal

probplot(data["sodium"], dist = "norm", plot=pylab)
# get sodium for hot cereals

hotCereals = data["sodium"][data["type"] == "H"]

# get sodium for cold cereals

coldCereals = data["sodium"][data["type"] == "C"]

# compare them

ttest_ind(hotCereals,coldCereals, equal_var=False)
print("mean sodium for hot cereals")

print(hotCereals.mean())

print("mean sodium for cold cereals")

print(coldCereals.mean())
# plot a histogram where each cereal type is a different color 

plt.hist(coldCereals, alpha = 0.5, label = "cold") 

plt.hist(hotCereals, label = "hot") 

plt.legend(loc = "upper right")

plt.title("Sodium(mg) content of cereals by type")

# Histograms using seaborn

sns.distplot(coldCereals, kde = False, label = "cold")

sns.distplot(hotCereals, kde = False, color="r", label = "hot")

plt.legend(loc = "upper right")
