# load in our libraries

import pandas as pd # pandas for data frames

from scipy.stats import ttest_ind # just the t-test from scipy.stats

from scipy.stats import probplot # for a qqplot

import matplotlib.pyplot as plt # for a qqplot
# read in our data

cereals = pd.read_csv('../input/cereal.csv')

# check out the first few lines

cereals.head()
# check normality by plotting a qqplot on a numerical feature of the dataset(using sodium). If the variable is normally distributed, most of the points 

# should be along the center diagonal.

%matplotlib inline

%pylab inline 

# I had to use the magic command in other for jupyter to plot the pylab)

probplot(cereals["sodium"], dist="norm", plot=pylab)
# Get the sodium for hot cereals

hotCereals = cereals['sodium'][cereals['type'] =='H']



# Get the sodium for cold cereals

coldCereals = cereals['sodium'][cereals['type'] == 'C']



#Compare them with t-test

ttest_ind(hotCereals, coldCereals, equal_var=False)

# let's look at the means (averages) of each group to see which is larger

print("Mean sodium for the hot cereals:")

print(hotCereals.mean())



print("Mean sodium for the cold cereals:")

print(coldCereals.mean())
# plot the cold cereals

# alpha makes the plots transparent

plt.hist(coldCereals, alpha=0.5, label='cold')

# and the hot cereals

plt.hist(hotCereals, label='hot')

# and add a legend

plt.legend(loc='upper right')

# add a title

plt.title("Sodium(mg) content of cereals by type (Cold or Hot)")

# label axis

plt.xlabel("Type (Cold or Hot)") # label the x axes 

plt.ylabel("Count") # label the y axes. In histogram it is usually called "Count".