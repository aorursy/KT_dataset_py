#importing important libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
#Loading the dataset
dataset = pd.read_csv("../input/haberman.csv",header=None,
        names=['age', 'year_of_operation', 'axillary_nodes', 'survival_status'])
#No. of Datapoints and features
dataset.shape
#Column names
dataset.columns
print(dataset.head(10))
#Counting number of datapoints in each class
dataset['survival_status'].value_counts()
dataset.describe()
#2-D scatter plots - Multivariate Analysis
sns.set_style("whitegrid")
sns.pairplot(dataset,hue='survival_status',vars=[dataset.columns[0],dataset.columns[1],
                                                 dataset.columns[2]],height=4)
#Univariate Analysis
#Plotting histogram along with PDF.
for i,attr in enumerate(list(dataset.columns)[:-1]):
    sns.FacetGrid(dataset,hue='survival_status',height=4).map(sns.distplot,attr).add_legend()
    plt.show()
#Plotting CDF of the survivors(after5)
after5 = dataset[dataset['survival_status']==1]
plt.figure(figsize=(15,8))
sns.set_style("whitegrid")
for i,attr in enumerate(list(dataset.columns)[:-1]):
    plt.subplot(1,3,i+1)
    print("---------",attr,"----------")
    counts,bin_edges = np.histogram(after5[attr],bins=10,density=True)
    print("Bin_Edges:- ",bin_edges)
    pdf = counts/sum(counts)
    print("PDF:- ",pdf)
    cdf = np.cumsum(pdf)
    print("CDF:- ",cdf)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel(attr)
#Plotting CDF of those who died with in 5 years
within5 = dataset[dataset['survival_status']==2]
plt.figure(figsize=(15,8))
sns.set_style("whitegrid")
for i,attr in enumerate(list(dataset.columns)[:-1]):
    plt.subplot(1,3,i+1)
    print("-------------",attr,"--------------")
    counts,bin_edges = np.histogram(within5[attr],bins=10,density=True)
    print("Bin_Edges:- ",bin_edges)
    pdf = counts/sum(counts)
    print("PDF:- ",pdf)
    cdf = np.cumsum(pdf)
    print("CDF:- ",cdf)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel(attr)
#Box plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i,attr in enumerate(list(dataset.columns)[:-1]):
    sns.boxplot(x='survival_status',y=attr,data=dataset, ax=axes[i])
plt.show()
#Calculating 25th,50th and 75th percentile of the survivors and those who died w.r.t to axillary nodes
print(np.percentile(after5['axillary_nodes'],(25,50,75)))
print(np.percentile(within5['axillary_nodes'],(25,50,75)))
#Violin Plot
fig,axes = plt.subplots(1,3,figsize=(15,5))
for i,attr in enumerate(list(dataset.columns)[:-1]):
    sns.violinplot(x='survival_status', y=attr, data = dataset, ax=axes[i])
plt.show()