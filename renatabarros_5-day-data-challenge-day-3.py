import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pylab

from scipy.stats import ttest_ind

from scipy.stats import probplot # for a qqplot



# Read the file into a dataset

cereals = pd.read_csv("../input/cereal.csv")

# Check first few rows

cereals.head()
# QQplot to check normality: if the variable is normally distributed most points will

# fall on centre diagonal.



# QQplots take your sample data, sort it in ascending order, and then plot them

# versus quantiles calculated from a theoretical distribution. The number of quantiles

# is selected to match the size of your sample data (Clay Ford, University of Virginia).



probplot(cereals["sodium"], dist='norm', plot=pylab)
# Na for hot cereals

a = cereals["sodium"][cereals["type"] == "C"]



# Na for cold cereals

b = cereals["sodium"][cereals["type"] == "H"]



ttest_ind(a, b, equal_var=False)



# Results: statistic = t, and pvalue

# A t-test will return a p-value. If a p-value is very low (generally below 0.01)

# this is evidence that it’s unlikely that we would have drawn our second sample

# from the same distribution as the first just by chance. 
# Compare the means of the two groups

print("Mean sodium for cold cereals:")

print(np.mean(a))

print("Mean sodium for hot cereals:")

print(np.mean(b))



# Plotting two histograms, one for each analysed group



plt.hist(a, alpha=0.5, label="Cold")

plt.hist(b, label="Hot")

plt.title("Sodium content in cereals")

plt.legend(loc="upper right")

plt.xlabel("Na (mg)")

plt.ylabel("Frequency")