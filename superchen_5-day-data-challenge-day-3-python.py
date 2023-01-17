

import pandas as pd

from scipy.stats import ttest_ind

from scipy.stats import probplot

import matplotlib.pyplot as plt

import pylab

cereal = pd.read_csv("../input/cereal.csv")

cereal.head()
# plot a qqplot to check normality. If the varaible is normally distributed, most of the points 

# should be along the center diagonal.

probplot(cereal["sodium"], dist="norm", plot=pylab)
hotCereal = cereal["sodium"][cereal["type"] == "H"]

coldCereal = cereal["sodium"][cereal["type"] == "C"]

ttest_ind(hotCereal, coldCereal, equal_var=False)
print("Mean the hot cereals:")

print(hotCereal.mean())



print("Mean for the cold cereals:")

print(coldCereal.mean())
plt.hist(coldCereal, alpha=0.5, label='cold')

plt.hist(hotCereal, label='hot')

plt.legend(loc='upper right')

plt.title("Sodium(mg) content of cereals by type")
print("I'm not quite familar with statistics. Maybe more practice :)")