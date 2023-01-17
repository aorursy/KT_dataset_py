import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind # just the t-test from scipy.stats
from scipy.stats import probplot # for a qqplot
import matplotlib.pyplot as plt # for a qqplot
import pylab # library similar to Matlab

# Change the plot style
plt.style.use('fivethirtyeight')
df = pd.read_csv("../input/cereal.csv")
df.head(20)
# This is copied from Rachael Tatman's notebook here:
# https://www.kaggle.com/rtatman/5-day-data-challenge-day-3-python

# plot a qqplot to check normality. If the varaible is normally distributed, most of the points 
# should be along the center diagonal.
probplot(df["sodium"], dist="norm", plot=pylab)
plt.show()
# extract hot and cold cereals from dataframe
hot_cereals = df["sodium"][df["type"] == 'H']
cold_cereals = df["sodium"][df["type"] == 'C']

print("In our dataset there are {} hot and {} cold cereals.".format(hot_cereals.shape[0], cold_cereals.shape[0]))
print(ttest_ind(hot_cereals, cold_cereals, equal_var=False))
# Calculate the mean
print("The average sodium in hot cereals is {} g.".format(np.mean(hot_cereals)))
print("The average sodium in cold cereals is {} g.".format(np.mean(cold_cereals)))

# The following code is copied from Rachael Tatman's Python notebook:
# https://www.kaggle.com/rtatman/5-day-data-challenge-day-3-python

# plot the cold cereals
plt.hist(cold_cereals, alpha=0.5, label='cold')

# and the hot cereals
plt.hist(hot_cereals, label='hot')

# and add a legend
plt.legend(loc='upper right')

# add a title
plt.title("Sodium(mg) content of cereals by type")

# show the plot
plt.show()