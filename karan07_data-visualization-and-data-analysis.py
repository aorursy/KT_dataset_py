import os
print(os.listdir('../input'))
# Importing Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Reading the data

habermans = pd.read_csv('../input/haberman.csv')
# Renaming the column names for better understanding

habermans.columns = ['Age', 'Year', 'Nodes', 'SurvivalStatus']
# Renaming the values in the survival status for pair plots

habermans["SurvivalStatus"] = habermans["SurvivalStatus"].map({1 : "Yes", 2 : "No"})
# After renaming

habermans
# No of datapoints and features

habermans.shape
# Columns

habermans.columns
# Datapoints per class

habermans['SurvivalStatus'].value_counts()
# Performing Univariate Analysis

# PDF - Probability Density Funtion for Age

sns.FacetGrid(habermans, hue='SurvivalStatus',size=5)\
    .map(sns.distplot, "Age")\
    .add_legend()
plt.ylabel('PDF')
plt.title("PDF for Age")
plt.show()
# PDF - Probability Density Funtion for Year

sns.FacetGrid(habermans, hue='SurvivalStatus',size=5)\
    .map(sns.distplot, "Year")\
    .add_legend()
plt.ylabel('PDF')
plt.title("PDF for Year")
plt.show()
# PDF - Probability Density Funtion for Nodes

sns.FacetGrid(habermans, hue='SurvivalStatus',size=5)\
    .map(sns.distplot, "Nodes")\
    .add_legend()
plt.ylabel('PDF')
plt.title("PDF for Nodes")
plt.show()
# CDF - Cumulative Density Function
# CDF for Age

fig, ax = plt.subplots()
pdf, bin_edges = np.histogram(habermans['Age'][habermans['SurvivalStatus'] == 'Yes'], bins = 10, density = True)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = 'Yes')
plt.plot(bin_edges[1:], cdf, label = 'Yes')

pdf, bin_edges = np.histogram(habermans['Age'][habermans['SurvivalStatus'] == 'No'], bins = 10, density = True)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = 'No')
plt.plot(bin_edges[1:], cdf, label = 'No')

plt.xlabel('Bin Edges')
plt.ylabel('PDF/CDF')
plt.title("CDF for Age")
plt.legend()
plt.show()
# CDF for Year

fig,ax = plt.subplots()
pdf, bin_edges = np.histogram(habermans['Year'][habermans['SurvivalStatus'] == 'Yes'], bins = 10, density = True)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = 'Yes')
plt.plot(bin_edges[1:], cdf, label = 'Yes')

pdf, bin_edges = np.histogram(habermans['Year'][habermans['SurvivalStatus'] == 'No'], bins = 10, density = True)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = 'No')
plt.plot(bin_edges[1:], cdf, label = 'No')

plt.xlabel('Bin Edges')
plt.ylabel('PDF/CDF')
plt.title('CDF for Year')
plt.legend()
plt.show()
# CDF for Nodes

fig, ax = plt.subplots()
pdf, bin_edges = np.histogram(habermans['Nodes'][habermans['SurvivalStatus'] == 'Yes'], bins = 10, density = True)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "Yes")
plt.plot(bin_edges[1:], cdf, label = "Yes")

pdf, bin_edges = np.histogram(habermans['Nodes'][habermans['SurvivalStatus'] == 'No'], bins = 10, density = True)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "No")
plt.plot(bin_edges[1:], cdf, label = "No")

plt.xlabel('Bin Edges')
plt.ylabel('PDF/CDF')
plt.title('CDF for Nodes')
plt.legend()
plt.show()
# Box Plot
# BP for Age

sns.boxplot(x = habermans['SurvivalStatus'], y = habermans['Age'], data = habermans)
plt.title('Box Plot for Age')
plt.show()
# BP for Year

sns.boxplot(x = habermans['SurvivalStatus'], y = habermans['Year'], data = habermans)
plt.title('Box Plot for Year')
plt.show()
# BP for Nodes

sns.boxplot(x = habermans['SurvivalStatus'], y = habermans['Nodes'], data = habermans)
plt.title('Box Plot for Nodes')
plt.show()
# Violin Plot for Age

sns.violinplot(x = habermans['SurvivalStatus'], y = habermans['Age'], data = habermans, size = 8)
plt.title('Violin Plot For Age')
plt.show()
# Violin Plot for Year

sns.violinplot(x = habermans['SurvivalStatus'], y = habermans['Year'], data = habermans, size = 8)
plt.title('Violin Plot For Year')
plt.show()
# Violin Plot for Nodes

sns.violinplot(x = habermans['SurvivalStatus'], y = habermans['Nodes'], data = habermans, size = 8)
plt.title('Violin Plot For Nodes')
plt.show()
# Bivariate analysis - Pair plot

sns.set_style("whitegrid")
sns.pairplot(habermans, hue = "SurvivalStatus", size = 5)
plt.show()
# Scatter Plot for Age and Year

sns.set_style('whitegrid')
sns.FacetGrid(habermans, hue = 'SurvivalStatus', size = 4)\
    .map(plt.scatter, "Age", "Year")\
    .add_legend()
plt.title('Scatter Plot for Age and Year')
plt.show()
# Scatter Plot for Age and Nodes

sns.set_style('whitegrid')
sns.FacetGrid(habermans, hue = 'SurvivalStatus', size = 4)\
    .map(plt.scatter, "Age", "Nodes")\
    .add_legend()
plt.title('Scatter Plot for Age and Nodes')
plt.show()
# Scatter Plot for Year and Nodes

sns.set_style('whitegrid')
sns.FacetGrid(habermans, hue = 'SurvivalStatus', size = 4)\
    .map(plt.scatter, "Year", "Nodes")\
    .add_legend()
plt.title('Scatter Plot for Year and Nodes')
plt.show()