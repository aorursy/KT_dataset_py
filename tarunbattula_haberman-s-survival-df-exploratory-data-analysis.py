





#import necessary packages 

import pandas as pd# Data analysis and manipulation

import numpy as np# Numerical operations

import seaborn as sns# Data visualization

import matplotlib.pyplot as plt# Data visualization

'''downlaod haberman.csv from https://www.kaggle.com/gilsousa/habermans-survival-data-set/version/1'''

#Load haberman.csv into a pandas dataFrame.

colnames = ['age', 'year', 'nodes', 'status']

hdf=pd.read_csv('../input/haberman.csv',header= None , names= colnames)

hdf.head()
#checking for Null Values

hdf.isnull().sum()
#This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage

hdf.info()
#Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values

#To know statistical summary of data

hdf.describe()
#To lnow How many data points for each class are present

hdf["status"].value_counts()
#PDF shows how many of points lies in the same interval.(smoothed form of histogram)

sns.FacetGrid(hdf,hue="status",size=5).map(sns.distplot,"age").add_legend();plt.ylabel("Density");plt.title("Distribution of age")

sns.FacetGrid(hdf,hue="status",size=5).map(sns.distplot,"year").add_legend();plt.ylabel("Density");plt.title("Distribution of year of operation ")

sns.FacetGrid(hdf,hue="status",size=5).map(sns.distplot,"nodes").add_legend();plt.ylabel("Density");plt.title("Distribution of positive axillary nodes detected ")

plt.show();
#CDF -it gives the area under the probability density function from minus infinity to x .

one = hdf.loc[hdf["status"] == 1]

two = hdf.loc[hdf["status"] == 2]





counts, bin_edges = np.histogram(one['age'], bins=10,density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)



counts, bin_edges = np.histogram(two['age'], bins=10,density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)



plt.title("pdf and cdf distribution of age")

plt.xlabel("age")

plt.ylabel("% of person's")

label =['PDF of Status One','CDF of Status One','PDF of Status Two','CDF of Status Two']

plt.legend(label)
counts, bin_edges = np.histogram(one['year'], bins=10,density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)



counts, bin_edges = np.histogram(two['year'], bins=10,density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)



plt.title("pdf and cdf distribution of year of operation ")

plt.xlabel("age")

plt.ylabel("% of person's")

label =['PDF of Status One','CDF of Status One','PDF of Status Two','CDF of Status Two']

plt.legend(label)
counts, bin_edges = np.histogram(one['nodes'], bins=10,density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)



counts, bin_edges = np.histogram(two['nodes'], bins=10,density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)



plt.title("pdf and cdf distribution of positive axillary nodes detected")

plt.xlabel("age")

plt.ylabel("% of person's")

label =['PDF of Status One','CDF of Status One','PDF of Status Two','CDF of Status Two']

plt.legend(label)
#Box Plot :In descriptive statistics, a boxplot is a method for graphically depicting groups of numerical data through their quartiles

sns.boxplot(x='status',y='age',data=hdf).set_title("Survival_status based on Age");plt.show()

sns.boxplot(x='status',y='year',data=hdf).set_title("Survival_status based on year of operation");plt.show()

sns.boxplot(x='status',y='nodes',data=hdf).set_title("Survival_status based on positive axillary nodes detected");plt.show()

#Violin Plot

# It is combination of box plot and histogram

sns.violinplot(x = "status", y = "age",  data = hdf).set_title("Survival_status based on Age");plt.show()

sns.violinplot(x = "status", y = "year",  data = hdf).set_title("Survival_status based on year of operation");plt.show()

sns.violinplot(x = "status", y = "nodes",  data = hdf).set_title("Survival_status based on positive axillary nodes detected");plt.show()

#Pair plot :Plot pairwise relationships in a dataset.

sns.set_style("whitegrid");

sns.pairplot(hdf,hue = "status", vars = ["age", "year", "nodes"],) #code source seaborn 0.9.0 documentation.

plt.show()

# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.