import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


haberman = pd.read_csv("../input/haberman.csv")

print(haberman)
# Number of data points and feature

print(haberman.shape)
#columns in Haberman dataset
print(haberman.columns)
# Data points for each class

haberman["status"].value_counts()
print(haberman.describe()) #show all the essential values
#PDF
sns.set_style("whitegrid")
sns.FacetGrid(haberman,hue = "status",size = 5) \
   .map(sns.distplot,"age") \
   .add_legend();
plt.show
sns.set_style("whitegrid")
sns.FacetGrid(haberman,hue = "status",size = 5) \
   .map(sns.distplot,"year") \
   .add_legend();
plt.show();

sns.set_style("whitegrid")
sns.FacetGrid(haberman,hue = "status",size = 5) \
   .map(sns.distplot,"nodes") \
   .add_legend();
plt.show();
#CDF
plt.figure(figsize = (10,5))
for idx, columns in enumerate(list(haberman.columns)[:-1]):
    plt.subplot(1,3,idx+1)
    print("------ ",columns,"------")
    counts,bin_edges = np.histogram(haberman[columns], bins = 10, density = True)

    pdf = counts/(sum(counts))
    print("PDF : ",pdf)
    print("Bin edges : ",bin_edges)
    cdf = np.cumsum(pdf)
    print("CDF : ",cdf)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel(columns)




for columns in list(haberman.columns)[:-1]:
    print("\nMedian for {} : ".format(columns))
    print(np.median(haberman[columns]))
    print("Quantiles for {} : ".format(columns))
    print(np.percentile(haberman[columns],np.arange(0,100,25)))
    
# Box plot
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
for idx, col in enumerate(list(haberman.columns)[:-1]):
    sns.boxplot( x='status', y=col, data= haberman, ax=axes[idx])
plt.show()
#Violin plot
fig , axes = plt.subplots(1,3, figsize = (10,5))
for idx ,col in enumerate(list(haberman.columns)[:-1]):
    sns.violinplot(x = 'status', y = col , data = haberman , ax = axes[idx])
    
plt.show()
sns.set_style("whitegrid")
sns.FacetGrid(haberman,hue = "status",size = 4).map(plt.scatter,"year","nodes") \
   .add_legend()
plt.show
sns.pairplot(haberman,hue = "status", size = 3)
plt.show()