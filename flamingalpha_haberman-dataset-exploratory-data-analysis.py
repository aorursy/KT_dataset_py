#importing the libraries

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels import robust
#inserting the dataset haberman.csv into a pandas dataframe

habermand=pd.read_csv("../input/haberman.csv")

#Number of Points and Features in the Dataset

print(habermand.shape)
#The Column names in the Dataset

print(habermand.columns)
#We now try to look as to how the dataset is by looking at the head of the csv file
habermand.head()
#As the column names are not mentioned properly in the dataset given, we will try to rename them
#We try to store the column names in a list and integrate it into the our haberman dataframe

column_name = ["age","operation_year","positive_axilliary_nodes","survival_status"]

habermand=pd.read_csv("../input/haberman.csv", header=None, names=column_name)

habermand.head()
habermand["survival_status"].value_counts()
#so we get to know that this dataset is an imbalanced dataset because the number of datapoints for the class attribute is 
#not the same.

#checking the breif description of the dataset
habermand.info()
#Describing the dataset
habermand.describe()
#plotting pdf for age
sns.FacetGrid(habermand, hue="survival_status", size=5)\
    .map(sns.distplot, "age")\
    .add_legend();
plt.show();
#plotting pdf for operation_year
sns.FacetGrid(habermand, hue="survival_status", size=5)\
    .map(sns.distplot, "operation_year")\
    .add_legend();
plt.show();
#plotting pdf for positive_axilliary_nodes
sns.FacetGrid(habermand, hue="survival_status", size=5)\
    .map(sns.distplot, "positive_axilliary_nodes")\
    .add_legend();
plt.show();
#for cdf we divide the survived people and the people who have died
survived = habermand.loc[habermand["survival_status"] == 1];
died = habermand.loc[habermand["survival_status"] == 2];
#plotting cdf for positive axilliary nodes

#for people who survived for more than five year
counts, bin_edges = np.histogram(survived['positive_axilliary_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label = "pdf more than 5")
plt.plot(bin_edges[1:], cdf, label = "cdf more than 5")

#for people who didnt survive for more than five year
counts, bin_edges = np.histogram(died['positive_axilliary_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = "pdf less than 5")
plt.plot(bin_edges[1:], cdf, label = "cdf less than 5")
plt.legend()
plt.xlabel("positive_axilliary_nodes")

plt.show();

#plotting cdf for  operation year

#for people who survived for more than five year
counts, bin_edges = np.histogram(survived['operation_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label = "pdf more than 5")
plt.plot(bin_edges[1:], cdf, label = "cdf more than 5")

#for people who didnt survive for more than five year
counts, bin_edges = np.histogram(died['operation_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = "pdf less than 5")
plt.plot(bin_edges[1:], cdf, label = "cdf less than 5")
plt.legend()
plt.xlabel("operation_year")

plt.show();

#plotting cdf for age

#for people who survived for more than five year
counts, bin_edges = np.histogram(survived['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label = "pdf more than 5")
plt.plot(bin_edges[1:], cdf, label = "cdf more than 5")

#for people who didnt survive for more than five year
counts, bin_edges = np.histogram(died['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = "pdf less than 5")
plt.plot(bin_edges[1:], cdf, label = "cdf less than 5")
plt.legend()
plt.xlabel("age")

plt.show();

#boxplot for age v survival status
sns.boxplot(x='survival_status',y='age', data=habermand)
plt.show()
#boxplot for operation_year v survival status
sns.boxplot(x='survival_status',y='operation_year', data=habermand)
plt.show()
#boxplot for positive axilliary nodes v survival status
sns.boxplot(x='survival_status',y='positive_axilliary_nodes', data=habermand)
plt.show()
#violin plot for survival status v year
sns.violinplot(x='survival_status',y='operation_year', data=habermand)
plt.show()
#violin plot for survival status v axiliary nodes
sns.violinplot(x='survival_status',y='positive_axilliary_nodes', data=habermand)
plt.show()
#violinplot for survival status and age
sns.violinplot(x='survival_status',y='age', data=habermand)
plt.show()
 #Finding the mean and standard deviation for age
print("Means:")
print(np.mean(survived["age"]))
print(np.mean(died["age"]))
print("\n")
print("Std-dev:");
print(np.std(survived["age"]))
print(np.std(died["age"]))
#Finding the mean and standard deviation for positive axilliary nodes
print("Means:")
print(np.mean(survived["positive_axilliary_nodes"]))
print(np.mean(died["positive_axilliary_nodes"]))

print("\nStd-dev:");
print(np.std(survived["positive_axilliary_nodes"]))
print(np.std(died["positive_axilliary_nodes"]))
#Finding the mean and standard deviation for operating_year
print("Means:")
print(np.mean(survived["operation_year"]))
print(np.mean(died["operation_year"]))

print("\nStd-dev:");
print(np.std(survived["operation_year"]))
print(np.std(died["operation_year"]))
#scatter plot for age v node
sns.set_style("whitegrid");
sns.FacetGrid(habermand, hue="survival_status", size=6) \
   .map(plt.scatter, "age", "positive_axilliary_nodes") \
   .add_legend();
plt.show();
#scatter plot for operation year v node
sns.set_style("whitegrid");
sns.FacetGrid(habermand, hue="survival_status", size=6) \
   .map(plt.scatter, "operation_year", "positive_axilliary_nodes") \
   .add_legend();
plt.show();
#scatter plot for age v node
sns.set_style("whitegrid");
sns.FacetGrid(habermand, hue="survival_status", size=6) \
   .map(plt.scatter, "age", "operation_year") \
   .add_legend();
plt.show();
plt.close();
sns.set_style("whitegrid");
sns.pairplot(habermand, hue="survival_status", vars = ['age','operation_year','positive_axilliary_nodes'],size=5);
plt.show()