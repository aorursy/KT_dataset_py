# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")

%matplotlib inline

iris = pd.read_csv('../input/iris/Iris.csv')
iris.shape
iris.head()
# Let's check if there in any inconsitency in the data set
iris = iris.drop('Id', axis=1)
iris.info()
# Let's see how many examples we have of each species
iris["Species"].value_counts()
# Let's plot a scatter plot of the Iris features
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='Setosa',  figsize= (10,6))
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='Virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length vs Width")
plt.show()
# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure

sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)
plt.show()
# Sepal Length using a Strip plot
sns.stripplot(y ='SepalLengthCm', x = 'Species', data =iris)
plt.show()
# We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
plt.show()
# A violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data are fatter, and sparser thiner in a violin plot

sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)
plt.show()
# Let's see how are the length and width are distributed
iris.hist(edgecolor='black',bins = 25, figsize= (12,6))
plt.show()
# Another useful seaborn plot is the pairplot, which shows the bivariate relation between each pair of features

# From the pairplot, we'll see that the Iris-setosa species is separataed from the other two across all feature combinations

sns.pairplot(data = iris, hue="Species", size=3)
plt.show()
# Plotting heat map
sns.heatmap(iris.corr(), cmap="YlGnBu", annot=True, fmt="f")
plt.show()
iris.describe()
iris['Species'].unique()
# Filtering by species
indices = iris['Species'] == 'Iris-setosa'
setosa = iris.loc[indices,:]
indices = iris['Species'] == 'Iris-versicolor'
versicolor = iris.loc[indices,:]
indices = iris['Species'] == 'Iris-virginica'
virginica = iris.loc[indices,:]

# Delete the species column from each dataframe as same species are present

del setosa['Species'], versicolor['Species'], virginica['Species']
# Visual EDA for individual species

setosa.plot(kind = 'hist', bins =50, range = (0,8), alpha = 0.3)
plt.title('Setosa Data Set')
plt.xlabel('[cm]')

versicolor.plot(kind = 'hist', bins =50, range = (0,8), alpha = 0.3)
plt.title('Versicolor Data Set')
plt.xlabel('[cm]')

virginica.plot(kind = 'hist', bins =50, range = (0,8), alpha = 0.3)
plt.title('Virginica Data Set')
plt.xlabel('[cm]')

plt.show()
# ECDF
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    n = len(data)

    x = np.sort(data)
    y = np.arange(1, n+1) / n

    return x, y

# Comparing ECDFs
x_set, y_set = ecdf(setosa['PetalLengthCm'])
x_vers, y_vers = ecdf(versicolor['PetalLengthCm'])
x_virg, y_virg = ecdf(virginica['PetalLengthCm'])


# Plot all ECDFs on the same plot
_ = plt.figure( figsize= (8,5))
_ = plt.plot(x_set, y_set, marker = '.', linestyle = 'none')
_ = plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')
_ = plt.plot(x_virg, y_virg, marker = '.', linestyle = 'none')

# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()