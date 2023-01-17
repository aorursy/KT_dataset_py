# Import required libraries for plotting



# pandas is an open source library 

# providing high easy-to-use data structures and data analysis tools

# data processing, CSV file I/O (e.g. pandas.read_csv)

import pandas as pandas



# Seaborn is a Python visualization library based on matplotlib. 

# It provides a high-level interface for drawing attractive statistical graphics.

# Current version of seaborn generates a bunch of warnings that we'll ignore

import warnings 

warnings.filterwarnings("ignore")

import seaborn as seaborn



# matplotlib is a python 2D plotting library

import matplotlib.pyplot as matplot

seaborn.set(style="white", color_codes=True)
# Load the admit data

# the dataset is now a Pandas DataFrame

admitData = pandas.read_csv("../input/SampleAdmitData_Train.csv") 
# See sample data in admitData DataFrame

admitData.head()
# Show number of rows in admitData

len(admitData)
# Describe the data

# Show non-null count, mean, std dev, min, max, etc

admitData.describe()
# See how many succcess or failures we have in admit column

admitData["admit"].value_counts()
# Do a simple scatter plot between gre and gpa

# We want to see if high gre corresponds to a high gpa

# This uses .plot extension of Pandas dataframe

admitData.plot(kind="scatter", x="gre", y="gpa", ylim=(None, 4.25), xlim=(None, 850))
# We can also use the seaborn library to make a similar plot

# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure

# size is the size of the output chart

seaborn.jointplot(x="gre", y="gpa", data=admitData, ylim=(2, 4.25), xlim=(250, 850))
# Add regression and kernel density fits to the chart above

seaborn.jointplot(x="gre", y="gpa", data=admitData, kind="reg", ylim=(None, 4.25), xlim=(None, 850))