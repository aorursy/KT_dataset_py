# import libraries
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind # just the t-test from scipy.stats
from scipy.stats import probplot # for a qqplot
import pylab # for the probability plot 

# read in and show the first few lines
cereal = pd.read_csv("../input/cereal.csv")

# We should make sure that the variable is normally distributed
# so let's use a qq-polt to do that

# plot a qqplot to check normality. If the variable is normally distributed, most of the points 
# should be along the center diagonal.
probplot(cereal["sodium"], dist="norm", plot=pylab)
# Preform our t-test

# get the sodium for hot ceareals
hotCereals = cereal["sodium"][cereal["type"] == "H"]

# get the sodium for cold ceareals
coldCereals = cereal["sodium"][cereal["type"] == "C"]

# compare them
ttest_ind(hotCereals, coldCereals, equal_var=False)
# let's look at the means (averages) of each group to see which is larger
print("Mean sodium for the hot cereals: {}".format(hotCereals.mean()))

print("Mean sodium for the cold cereals: {}".format(coldCereals.mean()))
# Now plot for the two cereal types, with each as a different color
# alpha sets that transparency of a plot. 0 = completely transparent (you won't see it), 1 = completely opaque (you can't see through it)
# I set it 0.5 here so that the values of the two plots will be different and it'll be easier to tell them apart

# plot the cold cereals
plt.hist(coldCereals, alpha=0.5, label='cold', edgecolor = "black")

# and the hot cereals
plt.hist(hotCereals, label='hot', edgecolor = "black")

# and add a legend
plt.legend(loc='upper right')

# add a title
plt.title("Sodium(mg) content of cereals by type")