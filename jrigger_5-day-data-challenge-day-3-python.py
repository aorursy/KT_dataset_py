# load in our libraries

import pandas as pd # pandas for data frames

from scipy.stats import ttest_ind # just the t-test from scipy.stats

from scipy.stats import probplot # for a qqplot

import matplotlib.pyplot as plt # for a qqplot

import pylab



# read the data

cerealData = pd.read_csv("../input/cereal.csv")

cerealData.head()

probplot(cerealData["sodium"], dist="norm", plot=pylab)
# get the sodium for hot cereals

hotCereals = cerealData["sodium"][cerealData["type"]=="H"]

# get the sodium for cold cereals

coldCereals = cerealData["sodium"][cerealData["type"]=="C"]



ttest_ind(hotCereals, coldCereals, equal_var=False)
print ("Mean sodium for the hot cereals is: %s" %hotCereals.mean())

print ("Mean sodium for the cold cereals is: %s" %coldCereals.mean())
plt.hist(coldCereals, alpha=0.5, label = "Cold")

plt.hist(hotCereals, label="Hot")

plt.legend(loc = "upper right")

plt.title("Sodium(mg) content of cereals by type")