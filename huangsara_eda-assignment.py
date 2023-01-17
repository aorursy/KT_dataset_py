# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import numpy as np # linear algebra
import warnings # current version of seaborn generates a bunch of warnings but for the purposes of this assignment, can ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results writen to the current directory are saved as output.
avoc = pd.read_csv("../input/avocado.csv") # the avocado dataset is now a Pandas DataFrame
avoc.head() # press shift+enter to execute this cell
# This lets us see how many entries there are for each region
# All but one of the regions contain 338 entries; WestTexNewMexico contains 335 (3 less than the others)
avoc["region"].value_counts()
# The first way we can plot things is using the .plot extension from Pandas dataframes
# We can use this to make a scatterplot of the avocado features (comparing Total volume against average price).
# With less total volume, the average price is higher, which makes sense when thinking about unit price.
avoc.plot(kind = "scatter", x = "Total Volume", y = "AveragePrice")
# These boxplots compare features
avoc.drop("Unnamed: 0", axis = 1).boxplot(by = "region", figsize = (12, 12))
# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
sns.jointplot(x = "Total Volume", y = "AveragePrice", data = avoc, size = 5)
# One piece of information missing in the scatterplot above is what region each avocado is from
# We'll use seaborn's FacetGrid to color the scatterplot by region so that we know where each data point on the scatterplot is coming from.
sns.FacetGrid(avoc, hue = "region", size = 5) \
   .map(plt.scatter, "Total Volume", "AveragePrice") \
   .add_legend()
# look at an individual feature in Seaborn through a boxplot
# This shows how the average price per avocado has changed per region through a boxplot
sns.boxplot(x = "region", y = "AveragePrice", data = avoc)
# One way to extend the above plot is adding a layer of individual points on top of
# it through Seaborn's stripplot
#
# The dots show the data points for each region in the box plot, showing the density
#
ax = sns.boxplot(x = "region", y = "AveragePrice", data = avoc)
ax = sns.stripplot(x = "region", y = "AveragePrice", data = avoc, jitter = True, edgecolor = "gray")
# A violin plot combines the above two plots in one compact form
# Denser regions of the data are fatter/wider, and sparser regions are slimmer/thinner in a violin plot
sns.violinplot(x = "region", y = "AveragePrice", data = avoc, size = 1000)
# Another seaborn plot that is useful for looking at univariate relations is the kdeplot
# This shows how the average price per avocado for each region most likely will be in terms of the density
sns.FacetGrid(avoc, hue = "region", size = 8) \
   .map(sns.kdeplot, "AveragePrice") \
   .add_legend()
# Another useful seaborn plot is the pairplot, which shows the bivariate relation
# between each pair of features
# 
# As mentioned in class, these plots help display useful information and relationships between features
sns.pairplot(avoc.drop("Unnamed: 0", axis = 1), hue = "region", size = 3)