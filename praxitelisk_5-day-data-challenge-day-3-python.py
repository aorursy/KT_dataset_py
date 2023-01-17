# load in our libraries
import pandas as pd # pandas for data frames
from scipy.stats import ttest_ind # just the t-test from scipy.stats
from scipy.stats import probplot # for a qqplot
import matplotlib.pyplot as plt # for a qqplot
import pylab #

# read in our data
cereals = pd.read_csv("../input/cereal.csv")
# check out the first few lines
cereals.head()
# plot a qqplot to check normality. If the varaible is normally distributed, most of the points 
# should be along the center diagonal.
probplot(cereals["sodium"], dist="norm", plot=pylab)
# get the sodium for hot cerals
hotCereals = cereals["sodium"][cereals["type"] == "H"]
# get the sodium for cold ceareals
coldCereals = cereals["sodium"][cereals["type"] == "C"]

# compare them
ttest_ind(hotCereals, coldCereals, equal_var=False)
# let's look at the means (averages) of each group to see which is larger
print("Mean sodium for the hot cereals:")
print(hotCereals.mean())

print("Mean sodium for the cold cereals:")
print(coldCereals.mean())
# plot the cold cereals
plt.hist(coldCereals, alpha=0.5, label='cold')
# and the hot cereals
plt.hist(hotCereals, label='hot')
# and add a legend
plt.legend(loc='upper right')
# add a title
plt.title("Sodium(mg) content of cereals by type")