import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../input/edahaberman/EDA-haberman.csv')
## Data dimensions
print(data.shape)
## Data columns 
print(data.columns)

## How many points of each class present
data["status"].value_counts()

data.plot(kind='scatter',x='age',y='nodes')
plt.show()
## 2D scatter plot for coloring each class
sns.set_style("whitegrid")
sns.FacetGrid(data,hue='status',height=4).map(plt.scatter,"age","nodes").add_legend()
plt.show()

plt.close()
sns.set_style("whitegrid")
sns.pairplot(data,hue='status',height=4)
plt.show()
one = data.loc[data["status"] == 1]
two = data.loc[data["status"] == 2]
plt.plot(one["age"], np.zeros_like(one["age"]), 'o', label = "status\n" "1")
plt.plot(two["age"], np.zeros_like(two["age"]), 'o', label = "2")
plt.title("1-D scatter plot for age")
plt.xlabel("age")
plt.legend()
plt.show()
sns.FacetGrid(data, hue="status",height=5).map(sns.distplot,"age").add_legend()
plt.ylabel("Density")
plt.title("Histogram of age")
plt.show()
sns.FacetGrid(data, hue="status",height=5).map(sns.distplot,"year").add_legend()
plt.ylabel("Density")
plt.title("Histogram of year")
plt.show()
sns.FacetGrid(data, hue="status",height=5).map(sns.distplot,"nodes").add_legend()
plt.ylabel("Density")
plt.title("Histogram of nodes")
plt.show()

plt.figure(figsize=(20,5))
for i, j in enumerate(list(data.columns)[:-1]):
    plt.subplot(1, 3, i+1)
    counts, bin_edges = np.histogram(data[j], bins=10, density=True)
    pdf = counts/sum(counts)
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
    plt.xlabel(j)
one = data.loc[data["status"] == 1]
two = data.loc[data["status"] == 2]
label = ["pdf of class 1", "cdf of class 1", "pdf of class 2", "cdf of class 2"]
counts, bin_edges = np.histogram(one["age"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.title("Age")
plt.xlabel("age")
plt.ylabel("% of person's")
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

counts, bin_edges = np.histogram(two["age"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(label)

plt.show()
label = ["pdf of class 1", "cdf of class 1", "pdf of class 2", "cdf of class 2"]
counts, bin_edges = np.histogram(one["year"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)



counts, bin_edges = np.histogram(two["year"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.title("pdf and cdf for year")
plt.xlabel("year")
plt.ylabel("% of person's")
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(label)

plt.show();
label = ["pdf of class 1", "cdf of class 1", "pdf of class 2", "cdf of class 2"]
counts, bin_edges = np.histogram(one["nodes"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

counts, bin_edges = np.histogram(two["nodes"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.title("pdf and cdf for nodes")
plt.xlabel("nodes")
plt.ylabel("% of person's")
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(label)

plt.show();
sns.boxplot(x='status',y='age',data=data)
plt.show()
sns.boxplot(x='status',y='year',data=data)
plt.show()
sns.boxplot(x='status',y='nodes',data=data)
plt.show()
sns.violinplot(x='status',y='age',data=data)
plt.show()
sns.violinplot(x='status',y='year',data=data)
plt.show()
sns.violinplot(x='status',y='nodes',data=data)
plt.show()