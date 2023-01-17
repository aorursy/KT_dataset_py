# This is the magic command. With this you don't have to import matplotlib.pyplot 

#%matplotlib inline



import pandas as pd # pandas for dataframes

import matplotlib.pyplot as plt

import pylab

from scipy.stats import ttest_ind  # just the ttest from scipy.stats

from scipy.stats import probplot  # for a qqplot 



# read in our data

cereals = pd.read_csv("../input/cereal.csv")



# check out the first few lines

cereals.head()





#scipy.stats.ttest_ind()
# plot a qqplot to check normality. If the variable is normally 

#  distributed most of the points should be along the center diagonal.

probplot(cereals["sodium"], dist="norm", plot=pylab)

# get the sodium for hot cereals

hotCereals = cereals["sodium"][cereals["type"] == "H"]

# get the sodium for cold cereals

coldCereals = cereals["sodium"][cereals["type"] == "C"]



# compare them

ttest_ind(hotCereals,coldCereals,equal_var=False)
# Let's look at the means (averages) of each groups to see which is larger

print("Mean sodium for the hot cereals:")

print(hotCereals.mean())

print("Mean sodium for the cold cereals:")

print(coldCereals.mean())
# plot the cold cereals 

plt.hist(coldCereals, label='cold')

# and the hot cereals 

plt.hist(hotCereals, label='hot')

# and add a legend 

plt.legend(loc='upper right')

# add a title

plt.title("Sodium content of cereals by type")
# get the potassium for hot cereals

hotCereals_potass = cereals["potass"][cereals["type"] == "H"]

# get the potassium for cold cereals

coldCereals_potass = cereals["potass"][cereals["type"] == "C"]



# compare them

ttest_ind(hotCereals_potass,coldCereals_potass,equal_var=False)
# Let's look at the means (averages) of each groups to see which is larger

print("Mean potassium for the hot cereals:")

print(hotCereals_potass.mean())

print("Mean potassium for the cold cereals:")

print(coldCereals_potass.mean())
plt.hist(coldCereals_potass, label='cold')

plt.hist(hotCereals_potass, label='hot')

plt.legend(loc='upper right')

plt.title("Potassium content of Cereals by Types")