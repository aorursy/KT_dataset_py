## Let me import the necessary libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
## Import the dataset

'''download haberman.csv from https://www.kaggle.com/gilsousa/habermans-survival-data-set'''

df = pd.read_csv("../input/haberman.csv")

df.head(5)
## Let us understand how big the dataset is! 

print (df.shape)
## Lets understand the type of data present in each column.

df.info()
## Lets look at the mean, Standard Deviation, Minimum, Maximum value in each row of the given Dataset.

df.describe()
## What are the Columns in the dataset 

print (df.columns)
## Print the number of people who survived more than 5 years and the number of people who couldnt

print(df["status"].value_counts())
## Calculate the Percent of Survivors and Deaths after the operation over a span of 5 years

percent_survived = (225/float(306))*100
print ("{}% of the People survived for more than 5 years after the operation." .format(percent_survived))

percent_not_survived = (81/float(306))*100
print ("{}% of the People failed to survive atleast 5 years after the operation." .format(percent_not_survived))
## Plot the Percent of Survivors and Deaths after the operation over a span of 5 years

plt.figure(figsize=(5,10))
sns.set_style("whitegrid");
sns.countplot(df["status"])

plt.title(" Number of patients Survived V/S Did not Survive ")

plt.show()
## Lets calculate How many unique cases of nodes were removed.

print(df["nodes"].value_counts())
## Plot the count plot for the number of nodes removed

plt.figure(figsize=(15,10))
sns.set_style("whitegrid");
sns.countplot(df["nodes"])

plt.title(" Number of nodes removed ")
plt.show()
## Calculate the percent of nodes removed

percent_nodes = (float(136)/306)*100
print("Nearly {}% of the patients did not remove any Axillary node".format(percent_nodes))
## Plot the number of patients operated per year

print(df["year"].value_counts())
## Plot the graph to visualise the number of patients treated per year.

plt.figure(figsize = (10,15))
sns.set_style("whitegrid")
sns.countplot(df["year"])
plt.title(" Number of patients treated per year")
plt.show()
## Plot the Histograms and Pdf curves with respect to 
## A) Age

sns.FacetGrid(df, hue="status", size=5) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.show();
## Plot the Histograms and Pdf curves with respect to 
## B) Year

sns.FacetGrid(df, hue="status", size=5) \
   .map(sns.distplot, "year") \
   .add_legend();
plt.show();
## Plot the Histograms and Pdf curves with respect to 
## C) Number of Axillary Nodes removed


sns.FacetGrid(df, hue="status", size=7) \
   .map(sns.distplot, "nodes") \
   .add_legend();
plt.show();
## Split the dataset into 2 sets, for Survived and Did not Survive separately

df_2_1 = df.loc[df["status"] == 1];
df_2_2 = df.loc[df["status"] == 2];
## Plots of CDF of Survival Status Based on Age.
## Survived Patients

plt.figure(figsize =(15,10))
plt.title("CDF with respect to Age")
plt.xlabel("Age of the Patient")
plt.ylabel("Percentile of Patients")
counts, bin_edges = np.histogram(df_2_1['age'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
print ("\n PDF : Status = 1 \n")
print(pdf);

print("\n Edges : Status = 1 \n")
print(bin_edges)

print ("\n ************************************************************************************** \n")
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


## Patients failed to survive
counts, bin_edges = np.histogram(df_2_2['age'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))

print ("\n PDF : Status = 2\n")
print(pdf);

print("\n Edges : Status = 2 \n")

print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

plt.show();
## Plots of CDF of Survival Status Based on Nodes Removed.
## Survived Patients

plt.figure(figsize =(10,5))
plt.title("CDF with respect to Nodes Removed")
plt.xlabel("Number of Nodes Removed")
plt.ylabel("Percentile of Patients")

counts, bin_edges = np.histogram(df_2_1['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print ("\n PDF : Status = 1 \n")
print(pdf);

print("\n Edges : Status = 1 \n")
print(bin_edges)

print ("\n ************************************************************************************** \n")
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


## Patients failed to survive
counts, bin_edges = np.histogram(df_2_2['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))

print ("\n PDF : Status = 2\n")
print(pdf);

print("\n Edges : Status = 2 \n")

print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

plt.show();
## Box plot for Status V/S Age 

sns.boxplot(x='status',y='age', data=df)
plt.show()
## Box plot for Status V/S Year of Treatment

sns.boxplot(x='status',y='year', data=df)
plt.show()
## Box plot for Status V/S Number of Nodes Removed.

sns.boxplot(x='status',y='nodes', data=df)
plt.show()
## Violin plot for Status V/S Age 

sns.violinplot(x='status',y='age', data=df)
plt.show()
## Violin plot for Status V/S Year of Treatment

sns.violinplot(x='status',y='year', data=df)
plt.show()
## Violin plot for Status V/S Number of Nodes Removed.

sns.violinplot(x='status',y='nodes', data=df)
plt.show()
## Plot the Pair Plots for the given datasets to find any relation between the  features

plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="status", size=4);
plt.show()