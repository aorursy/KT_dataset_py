# Required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set(style="ticks")
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")

# read the Iris dataset
iris_data = pd.read_csv("../input/Iris.csv") # data is in the pandas data frame format
iris_data.head(10) # checking the top 10 records
# Sumarize the Iris dataset
iris_data.describe()
# Group the Iris data set by "Species"
grouped = iris_data.groupby("Species")

# get the grouped data by species 
Iris_setosa = grouped.get_group("Iris-setosa")
Iris_versicolor = grouped.get_group("Iris-versicolor")
Iris_virginica= grouped.get_group("Iris-virginica")

# plot a "PetalLengthCm vs PetalWidthCm" for groups of species
fig, ax = plt.subplots()
ax.plot(Iris_setosa["PetalLengthCm"], Iris_setosa["PetalWidthCm"], 'bD', label = "Iris-setosa")
ax.plot(Iris_versicolor["PetalLengthCm"], Iris_versicolor["PetalWidthCm"],'b*',label = "Iris-versicolor" )
ax.plot(Iris_virginica["PetalLengthCm"], Iris_virginica["PetalWidthCm"],'bo', label = "Iris-virginica" )
ax.set_xlabel("PetalLengthCm")
ax.set_ylabel("PetalWidthCm")
ax.set_title("PetalLengthCm vs PetalWidthCm")

# Now add the legend with some customizations.
legend = ax.legend(loc='upper Left', shadow=True)

# check the grouped data
print ( "SUM OF SepalLengthCm , SepalWidthCm , PetalLengthCm , PetalWidthCm BY GROUP\n\n", grouped.sum())
print ("\n\nNUMBER OF ENTRIES IN EACH GROUP\n", grouped.count())

# plot "SepalLengthCm vs SepalWidthCm" for groups of species
grouped =  iris_data.groupby("Species")

fig, ax = plt.subplots()
for name, group in grouped:
    ax.plot(group.SepalLengthCm, group.SepalWidthCm, marker = "o",linestyle = "",ms=12, label = name)
ax.legend()
ax.set_xlabel("SepalLengthCm")
ax.set_ylabel("SepalWidthCm")
ax.set_title("SepalLengthCm vs SepalWidthCm")
# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
# Hexbin plot with marginal distributions

sns.jointplot(x = "PetalLengthCm", y ="PetalWidthCm", data=iris_data, size=5, kind="hex", color="#4CB391")
# We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Species", y="PetalLengthCm", data=iris_data)

