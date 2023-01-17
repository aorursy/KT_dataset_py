%matplotlib inline
# Import Pandas and then, since they are use frequently, import Series and Data Frame
# into the local namespace.
import pandas as pd
from pandas import Series, DataFrame

# Import matplotlib's pyplot to plot the data.
import matplotlib.pyplot as plt

# Load the Iris dataset which is in .csv format as a DataFrame. 
irisdf = pd.read_csv("../input/Iris.csv")

# Check the first records of irisdf.
irisdf.head()
# Check how many species there are.
irisdf["Species"].describe()
# See the frequencies for each species.
irisdf["Species"].value_counts()
# Summaryze the whole data frame (descriptive statistics on each column).
irisdf.describe()
# Scatter plot of a pair of features.
irisdf.plot(kind="scatter", x="SepalLengthCm", y="PetalLengthCm")
# Color/label dots according to Species 
ax = irisdf[irisdf.Species == "Iris-setosa"].plot(kind="scatter", x="SepalLengthCm", y="PetalLengthCm", color="red", label="Iris-setosa", title="SepalLenght vs. PetalLength")
irisdf[irisdf.Species == "Iris-virginica"].plot(kind="scatter", x="SepalLengthCm", y="PetalLengthCm", color="green", label="Iris-virginica", ax=ax)
irisdf[irisdf.Species == "Iris-versicolor"].plot(kind="scatter", x="SepalLengthCm", y="PetalLengthCm", color="blue", label="Iris-versicolor", ax=ax)

# Plot another pair of features
ax = irisdf[irisdf.Species == "Iris-setosa"].plot(kind="scatter", x="SepalWidthCm", y="PetalWidthCm", color="red", label="Iris-setosa", title="SepalWidth vs. PetalWidth")
irisdf[irisdf.Species == "Iris-virginica"].plot(kind="scatter", x="SepalWidthCm", y="PetalWidthCm", color="green", label="Iris-virginica", ax=ax)
irisdf[irisdf.Species == "Iris-versicolor"].plot(kind="scatter", x="SepalWidthCm", y="PetalWidthCm", color="blue", label="Iris-versicolor", ax=ax)
# Plot all combinations of two features scatter matrix
# First import panda's scatter_matrix into the namespace
from pandas.tools.plotting import scatter_matrix

# Plot by calling scatter_matrix
scatter_matrix(irisdf.drop("Id", axis=1), alpha=0.2, figsize=(10, 10), diagonal='kde')


# Pandas's scatter_matrix cannot distinguish categories by color, but the seaborn 
# pairplot can.
# Import seaborn
import seaborn as sbn

# Make a pairplot
sbn.pairplot(irisdf.drop("Id", axis=1), hue="Species", diag_kind="kde", size=3)
