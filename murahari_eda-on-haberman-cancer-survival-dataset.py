#Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os as os
import seaborn as sns
# Load data set
haberman = pd.read_csv("../input/haberman.csv")
# Get the top 5 data elements with header
haberman.head()
# As there is no column names as part of data so assign column names for it manually
haberman.columns = ["Age", "Year of operation", "axillary nodes", "Survival status"]
haberman.columns
# Get the top 5 data elements with header
haberman.head()
# Get the columns
haberman.columns
# Get how many data elements and columns are there
haberman.shape
# Get  data type of variables
haberman.dtypes
# Check is there any null values
haberman.isnull().values.any()
# Get total number of data elemnts who survived and not survived i.e, get total number of classifications for predictor 
haberman["Survival status"].value_counts()
# Plot a scatterplot for Age and  Year of operation
haberman.plot(kind="scatter", x="Age", y="Year of operation")
plt.show()
# Plot a scatterplot for Age and  axillary nodes
haberman.plot(kind="scatter", x="Age", y="axillary nodes")
plt.show()
# Using seaborn plot a scatterplot
sns.FacetGrid(data=haberman, hue="Survival status", size=5) \
    .map(plt.scatter, "Age", "axillary nodes") \
    .add_legend()
    
plt.show()
# Plotting pairplot for Class variable 'Sirvival status' with "Age", "Year of operation", "axillary nodes" as x and y axises
plt.close()
sns.pairplot(haberman, hue="Survival status", vars=["Age", "Year of operation", "axillary nodes"], size=3)
plt.show()
# Perform univariate analysis 
def density_plot(feature_var, class_var):
    '''Method to constuct a density plot with the given feature and class variables'''
    sns.set_style(style="whitegrid")
    sns.FacetGrid(data=haberman, hue=class_var, size=5) \
    .map(sns.distplot, feature_var) \
    .add_legend()

density_plot("Age", "Survival status")
plt.show()
density_plot("Year of operation", "Survival status")
plt.show()
density_plot("axillary nodes", "Survival status")
plt.show()
# Get the data elements having Survival status is 1
Haberman_Survived = haberman[haberman["Survival status"] == 1]
# Get the data elements having Survival status is 2
Haberman_Non_Survived = haberman[haberman["Survival status"] == 2]
#Get the counts and bin edges of axillary nodes whose survival status is 1
counts, bin_edges = np.histogram(Haberman_Survived["axillary nodes"], bins=30, density=True)
print (counts)
print (bin_edges)

#Get the counts and bin edges of axillary nodes whose survival status is 2
counts_Non, bin_edges_Non = np.histogram(Haberman_Non_Survived["axillary nodes"], bins=30, density=True)
print (counts_Non)
print (bin_edges_Non)

#PDF and CDF of survived
pdf_axillary_survived = counts/sum(counts)
cdf_axillary_survived = np.cumsum(pdf_axillary_survived)

#PDFand CDF of non survived
pdf_axillary_Non_survived = counts_Non/sum(counts_Non)
cdf_axillary_Non_survived = np.cumsum(pdf_axillary_Non_survived)

# Plot survived and non survived PDF, CDF in a single plot
plt.plot(bin_edges[1:], pdf_axillary_survived)
plt.plot(bin_edges[1:], cdf_axillary_survived)
plt.plot(bin_edges_Non[1:], pdf_axillary_Non_survived)
plt.plot(bin_edges_Non[1:], cdf_axillary_Non_survived)
#plt.xticks(np.linspace(0,50,13))
plt.xlabel("Axillary Node")

plt.legend(["Survived more than 5 years PDF", "Survived more than 5 years CDF", "Survived not more than 5 years PDF", "Survived not more than 5 years CDF" ])
plt.show()
# Get statistical description of the data elements whose Survival status is 1
haberman_Non_Survived = haberman[haberman["Survival status"] == 1]
print ("Summary of patients who are survived more than 5 yeras")
haberman_Non_Survived.describe()
# Get statistical description of the data elements whose Survival status is 2
haberman_Non_Survived = haberman[haberman["Survival status"] == 2]
print ("Summary of patients who are not survived more than 5 yeras")
haberman_Non_Survived.describe()
sns.boxplot(data=haberman, x="Survival status", y="Age")
plt.show()
sns.boxplot(data=haberman, x="Survival status", y="Year of operation")
plt.show()
sns.boxplot(data=haberman, x="Survival status", y="axillary nodes")
plt.show()
sns.violinplot(data=haberman, x="Survival status", y="Age")
plt.show()
sns.violinplot(data=haberman, x="Survival status", y="Year of operation")
plt.show()
sns.violinplot(data=haberman, x="Survival status", y="axillary nodes")
plt.show()
# Joint plot
sns.jointplot(data=haberman, x="Age", y="Year of operation", kind="kde")
plt.show()