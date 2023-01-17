# here we are importing important libraries which we are going to use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# here we are importing libraries for removing warning message
import warnings
warnings.filterwarnings("ignore")

# here we are reading our dataset
Data = pd.read_csv("../input/iriscsv/datasets_19_420_Iris.csv")
# here we are printing first five line of dataset
Data.head()
# here firstly we are checking if there is any NaN value or not
Data.isnull().sum()
# here we are checking shape of our dataset
Data.shape
# here we are printing info of our dataset
Data.info()
# here we are printing name of all the columns
Data.columns.values
# here we are showing how many examples we have of each species
Data["Species"].value_counts()
# Here we are using scatterplot for visualizing the Iris features.
plt.rcParams['figure.figsize'] = (10,5)
plt.style.use('dark_background')
Data.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
plt.title('Iris Feature', fontweight = 30, fontsize = 20)
plt.show()
# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=Data, size=5, kind= "resid")
plt.show()
# One piece of information missing in the plots above is what species each plant is
# We'll use seaborn's FacetGrid to color the scatterplot by species
sns.set(style="darkgrid", color_codes=True)
sns.FacetGrid(Data, hue="Species", size=5) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
   .add_legend()
plt.show()
# We can look at an individual feature in Seaborn through a boxplot
sns.set(style="dark", color_codes=True)
sns.barplot(x="Species", y="PetalLengthCm", data=Data)
plt.show()
# there is One way we can extend this plot is adding a layer of individual points on top of
# it through Seaborn's striplot
sns.set(style="white", color_codes=True)
ax = sns.boxplot(x="Species", y="PetalLengthCm", data=Data)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=Data, jitter=True, edgecolor="gray")
# here we are plotting violinplot as it give benifit by of the previous two plots and simplifies them
plt.style.use('bmh')
sns.violinplot(x="Species", y="PetalLengthCm", data=Data, size=6)
plt.show()
# frstly we will drop id column from our dataset
Data = Data.drop("Id",axis=1)
# Another useful seaborn plot is the pairplot, which shows the bivariate relation
# between each pair of features
# # From the pairplot, we'll see that the Iris-setosa species is separataed from the other
# two across all feature combinations
# here we are doing with regression
plt.style.use('fivethirtyeight')
sns.pairplot(Data, kind="reg",hue="Species", size=3)
plt.show()

# here we are ploting same graph as above but withour regression
plt.style.use('seaborn-dark-palette')
sns.pairplot(Data, kind="scatter",hue="Species", size=3)
plt.show()

# here we are ploting same above graph but with histogram
plt.style.use('classic')
sns.pairplot(Data, diag_kind="hist")
plt.show()

# Now we've covered seaborn, let's go back to some of the ones we can make with Pandas
# We can quickly make a boxplot with Pandas on each feature split out by species.
Data.boxplot(by="Species", figsize=(12, 6))
plt.show()
# One cool more sophisticated technique pandas has available is called Andrews Curves
# Andrews Curves involve using attributes of samples as coefficients for Fourier series
# and then plotting these
# here we are importing library for making andrews curves
plt.rcParams['figure.figsize'] = (10,5)
plt.style.use('bmh')
pd.plotting.andrews_curves(Data, 'Species')
plt.xlabel("species")
plt.show()

# Another multivariate visualization technique pandas has is parallel_coordinates
# Parallel coordinates plots each feature on a separate column & then draws lines
# connecting the features for each data sample
plt.style.use('classic')
pd.plotting.parallel_coordinates(Data, "Species")
plt.xlabel("species")
plt.show()
# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
sns.set(style="white", color_codes=True)
pd.plotting.radviz(Data, "Species")
plt.show()