import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt 



#day 3

from scipy.stats import ttest_ind 

from scipy.stats import probplot

import pylab 
cereals = pd.read_csv("../input/cereal.csv")

cereals.head()
cereals.describe()
cereal = "sodium"

cereal = "sugars"
probplot(cereals[cereal], dist="norm", plot=pylab)

hotCereals = cereals[cereal][cereals["type"] == "H"]

coldCereals = cereals[cereal][cereals["type"] == "C"]

ttest_ind(hotCereals, coldCereals, equal_var=False)
print("Mean {} Hot : {}".format(cereal, hotCereals.mean()))

print("Mean {} Cold : {}".format(cereal, coldCereals.mean()))
plt.hist(coldCereals, alpha=0.5, label='cold')

plt.hist(hotCereals, label='hot')

plt.legend(loc='upper right')

plt.title(cereal)
# Category Type - Hot / Cold 
