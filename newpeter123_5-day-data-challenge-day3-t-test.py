import pandas as pd

from scipy.stats import ttest_ind

from scipy.stats import probplot

import matplotlib.pyplot as plt



# read in our data

cereals = pd.read_csv("../input/cereal.csv")

# check out the first few lines

cereals.head()
# plot a qqplot to check normality. If the variable is normally distributed, most of the points should be along the center diagonal.

probplot(cereals["sodium"], dist="norm", plot=pylab)
# The second square bracket is the boolean filter for the main df

# get sodioum for hot cerials

hotCereals = cereals["sodium"][cereals["type"] == "H"]

# get sodium for cold ceareals

coldCereals = cereals["sodium"][cereals["type"] == "C"]

# compare them

ttest_ind(hotCereals, coldCereals,equal_var=False)
# want to see if the means are different



print("Mean sodium for the hot cereals:",hotCereals.mean())

print("Mean sodium for the cold cereals:", coldCereals.mean())

#Lets plot some graphs as above make the cold and hot diff colors



plt.hist(hotCereals, alpha=0.5, label='hot')

plt.hist(coldCereals, alpha=0.5, label='cold')

plt.legend(loc='upper right')
# Same as above 



plt.hist(hotCereals, alpha=1, label='hot')

plt.hist(coldCereals, alpha=0.2, label='cold')

plt.legend(loc='upper right')

plt.title("Sodum(mg) content of cereals by type")
