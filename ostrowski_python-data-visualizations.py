# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
import numpy as np

# Next, we'll load the Iris flower dataset, which is in the "../input/" directory
iris = pd.read_csv("../input/Iris.csv") # the iris dataset is now a Pandas DataFrame
# Our data is loaded as iris
# Let's see what's in the iris data executing iris.head() python comand
# Note that the first fours columns of data we have Quantitative data
# Only the last column 'Species' is Qualitative data
iris.head()
# Let's see how many examples we have of each species
# This question make us to think in Qualitative data in terms of Quantitative!
iris["Species"].value_counts()
# From the values above it seems we have only three types of species
# Let's make sure using unique values
print('This data set has {} different species: {}'
      .format(iris['Species'].nunique(), list(iris['Species'].unique())))
# Again plotting qualitative data in terms of quantitative
# Note that plotting make the short cut presenting you the unique values automaticly
sns.countplot(x='Species', data=iris);
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=8, stat_func=None);
# Categorical scatter plot by Species
sns.FacetGrid(iris, hue="Species", size=5) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
    .add_legend();
# Plotting in different columns using support linear regressions
sns.lmplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, col='Species');





