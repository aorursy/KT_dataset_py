import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib import gridspec
os.getcwd()
haberman = pd.read_csv("../input/haberman.csv", names = ['Age' , 'Op_Year','Axil_Nodes', 'Survival_Status'])
haberman.head()
print("Size of dataset is: ",haberman.shape)
print("Columns titles of dataset is: ",haberman.columns)
haberman['Survival_Status'].value_counts()

haberman.describe()
haberman['Survival_Status'].value_counts(normalize = True)
Sur_true = haberman.loc[haberman["Survival_Status"] == 1]
Sur_false = haberman.loc[haberman["Survival_Status"] == 2]

fig = plt.figure(figsize = (18,5))
plt.title("Survival Status")
plt.subplot(1,3,1)
plt.plot(Sur_true["Age"],np.zeros_like(Sur_true["Age"]),'o', label = "Survived above 5 Years")
plt.plot(Sur_false["Age"],np.zeros_like(Sur_false["Age"]),'^', label = "Died below 5 Years")
plt.xlabel("Age")
plt.legend()

plt.subplot(1,3,2)
plt.plot(Sur_true["Axil_Nodes"],np.zeros_like(Sur_true["Axil_Nodes"]),'o', label = "Survived above 5 Years")
plt.plot(Sur_false["Axil_Nodes"],np.zeros_like(Sur_false["Axil_Nodes"]),'^', label = "Died below 5 Years")
plt.xlabel("Axil_Nodes")
plt.legend()

plt.subplot(1,3,3)
plt.plot(Sur_true["Op_Year"],np.zeros_like(Sur_true["Op_Year"]),'o', label = "Survived above 5 Years")
plt.plot(Sur_false["Op_Year"],np.zeros_like(Sur_false["Op_Year"]),'^', label = "Died below 5 Years")
plt.xlabel("Op_Year")
plt.legend()
plt.show()
sns.FacetGrid(haberman, hue = 'Survival_Status').map(sns.distplot, 'Age').add_legend()
sns.FacetGrid(haberman, hue = 'Survival_Status').map(sns.distplot, 'Op_Year').add_legend()
sns.FacetGrid(haberman, hue = 'Survival_Status').map(sns.distplot, 'Axil_Nodes').add_legend()
plt.show()
#Survived after 5 years

counts , bin_edges = np.histogram(Sur_true["Axil_Nodes"], bins = 10, density = True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)

fig = plt.figure(figsize=(14, 3)) 
plt.subplot(1,3,1)
plt.plot(bin_edges[1:], pdf, label = "pdf")
plt.plot(bin_edges[1:], cdf, label = "cdf")
plt.xlabel("Axil_Nodes")
plt.legend()

plt.subplot(1,3,2)
counts , bin_edges = np.histogram(Sur_true["Age"], bins = 10, density = True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "pdf")
plt.plot(bin_edges[1:], cdf, label = "cdf")
plt.xlabel("Age")
plt.legend()


counts , bin_edges = np.histogram(Sur_true["Op_Year"], bins = 10, density = True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.subplot(1,3,3)
plt.plot(bin_edges[1:], pdf, label = "pdf")
plt.plot(bin_edges[1:], cdf, label = "cdf")
plt.xlabel("Op_Year")
plt.legend()
plt.tight_layout()
plt.show()

print("Quantiles for Axillary Nodes: ",np.percentile(Sur_true["Axil_Nodes"],np.arange(0,100,25)))
print("Quantiles for Age:",np.percentile(Sur_true["Age"],np.arange(0,100,25)))
print("90th Percentile for Axilallary Nodes",np.percentile(Sur_true["Axil_Nodes"],90))
print("90th Percentile for Age",np.percentile(Sur_true["Age"],90))
#Died within 5 years

counts , bin_edges = np.histogram(Sur_false["Axil_Nodes"], bins = 10, density = True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
fig = plt.figure(figsize=(14, 3)) 
plt.subplot(1,3,1)
plt.plot(bin_edges[1:], pdf, label = "pdf")
plt.plot(bin_edges[1:], cdf, label = "cdf")
plt.xlabel("Axil_Nodes")
plt.legend()


counts , bin_edges = np.histogram(Sur_false["Age"], bins = 10, density = True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.subplot(1,3,2)
plt.plot(bin_edges[1:], pdf, label = "pdf")
plt.plot(bin_edges[1:], cdf, label = "cdf")
plt.xlabel("Age")
plt.legend()


counts , bin_edges = np.histogram(Sur_false["Op_Year"], bins = 10, density = True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.subplot(1,3,3)
plt.plot(bin_edges[1:], pdf, label = "pdf")
plt.plot(bin_edges[1:], cdf, label = "cdf")
plt.xlabel("Op_Year")
plt.legend()

plt.tight_layout()
plt.show()

print("Quantiles for Axillary Nodes: ",np.percentile(Sur_false["Axil_Nodes"],np.arange(0,100,25)))
print("Quantiles for Age:",np.percentile(Sur_false["Age"],np.arange(0,100,25)))
print("90th Percentile for Axilallary Nodes",np.percentile(Sur_false["Axil_Nodes"],90))
print("90th Percentile for Age",np.percentile(Sur_false["Age"],90))
plt.close()
fig = plt.figure(figsize=(14, 3)) 
plt.subplot(1,3,1)
sns.boxplot(x = 'Survival_Status', y = 'Age', data = haberman)
#plt.show()
plt.subplot(1,3,2)
sns.boxplot(x = 'Survival_Status', y = 'Axil_Nodes', data = haberman)
#plt.show()
plt.subplot(1,3,3)
sns.boxplot(x = 'Survival_Status', y = 'Op_Year', data = haberman)
plt.tight_layout()
plt.legend()
plt.show()
plt.close()
fig = plt.figure(figsize=(14, 3)) 

plt.subplot(1,3,1)
sns.violinplot(x = 'Survival_Status', y = 'Age', data = haberman)

plt.subplot(1,3,2)
sns.violinplot(x = 'Survival_Status', y = 'Axil_Nodes', data = haberman)

plt.subplot(1,3,3)
sns.violinplot(x = 'Survival_Status', y = 'Op_Year', data = haberman)

plt.tight_layout()
plt.show()
plt.close()
fig = plt.figure(figsize=(14,6))
sns.set_style("whitegrid")

#plt.subplot(1,3,1)
sns.FacetGrid(haberman, hue = "Survival_Status").map(plt.scatter, "Age", "Axil_Nodes")

#plt.subplot(1,3,2)
sns.FacetGrid(haberman, hue = "Survival_Status").map(plt.scatter, "Op_Year", "Axil_Nodes")

#plt.subplot(1,3,3)
sns.FacetGrid(haberman, hue = "Survival_Status").map(plt.scatter, "Age", "Op_Year")
plt.show()
plt.close()
sns.set_style("whitegrid")
sns.pairplot(haberman, hue="Survival_Status", vars = ["Age", "Op_Year", "Axil_Nodes"], height=3)
plt.show()