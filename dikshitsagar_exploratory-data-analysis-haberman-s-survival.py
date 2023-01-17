import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels import robust
%matplotlib inline
df = pd.read_csv('../input/habermans-survival-data-set/haberman.csv',header=None, names=['age', 'year', 'nodes', 'status'])
df.head()
df.info()
df.describe()
# How many data point and features:
df.shape
# What are the column names :
df.columns
# count of data points for each class in 'status':
df['status'].value_counts()
sns.distplot(df['status']).set_title('Distribution of Status')
sns.distplot(df['age']).set_title('Distribution of age')
sns.distplot(df['year']).set_title('Distribution of year')
sns.distplot(df['nodes']).set_title('Distribution of nodes')
sns.FacetGrid(df, hue = 'status',height = 6).map(sns.distplot,"age").add_legend()
plt.ylabel("Density")
plt.title('Survival Status vs Age')
sns.FacetGrid(df, hue = 'status',height = 6).map(sns.distplot,"nodes").add_legend()
plt.ylabel("Density")
plt.title('Survival Status vs Auxillary Nodes')
sns.FacetGrid(df, hue = 'status',height = 6).map(sns.distplot,"year").add_legend()
plt.ylabel("Density")
plt.title('Survival Status vs Operation_year')
# Plot CDF for 'Age'
counts, bin_edges = np.histogram(df['age'], bins = 10, density = True)

pdf_age = counts/sum(counts)
print(pdf_age)
print(bin_edges)
cdf_age = np.cumsum(pdf_age)
print(cdf_age)
plt.figure(figsize=(9,6))
plt.plot(bin_edges[1:],pdf_age)
plt.plot(bin_edges[1:],cdf_age)
plt.ylabel('Density')
plt.xlabel('Age')
plt.legend(['PDF of Age','CDF of Age'])
plt.show()
# creating data frame for each status
Survived = df.loc[df["status"] == 1]
Not_Survived = df.loc[df["status"] == 2]

counts_S, bin_edges_S = np.histogram(Survived['age'], bins = 10, density = True)

pdf_age_survived = counts_S/sum(counts_S)
cdf_age_survived = np.cumsum(pdf_age_survived)


counts_NS, bin_edges_NS = np.histogram(Not_Survived['age'], bins = 10, density = True)

pdf_age_Not_survived = counts_NS/sum(counts_NS)
cdf_age_Not_survived = np.cumsum(pdf_age_Not_survived)

plt.figure(figsize=(9,6))
plt.plot(bin_edges_S[1:],pdf_age_survived)
plt.plot(bin_edges_S[1:],cdf_age_survived)

plt.plot(bin_edges_NS[1:],pdf_age_Not_survived)
plt.plot(bin_edges_NS[1:],cdf_age_Not_survived)


plt.ylabel('Density')
plt.xlabel('Age')
plt.legend(['PDF of Age Survived','CDF of Age Survived','PDF of Age Not Survived','CDF of Age Not Survived'])
plt.show()
counts_S, bin_edges_S = np.histogram(Survived['year'], bins = 10, density = True)

pdf_year_survived = counts_S/sum(counts_S)
cdf_year_survived = np.cumsum(pdf_year_survived)


counts_NS, bin_edges_NS = np.histogram(Not_Survived['year'], bins = 10, density = True)

pdf_year_Not_survived = counts_NS/sum(counts_NS)
cdf_year_Not_survived = np.cumsum(pdf_year_Not_survived)

plt.figure(figsize=(9,6))
plt.plot(bin_edges_S[1:],pdf_year_survived)
plt.plot(bin_edges_S[1:],cdf_year_survived)

plt.plot(bin_edges_NS[1:],pdf_year_Not_survived)
plt.plot(bin_edges_NS[1:],cdf_year_Not_survived)


plt.ylabel('Density')
plt.xlabel('year')
plt.legend(['PDF of year Survived','CDF of year Survived','PDF of year Not Survived','CDF of year Not Survived'])
plt.show()
counts_S, bin_edges_S = np.histogram(Survived['nodes'], bins = 10, density = True)

pdf_nodes_survived = counts_S/sum(counts_S)
cdf_nodes_survived = np.cumsum(pdf_nodes_survived)


counts_NS, bin_edges_NS = np.histogram(Not_Survived['nodes'], bins = 10, density = True)

pdf_nodes_Not_survived = counts_NS/sum(counts_NS)
cdf_nodes_Not_survived = np.cumsum(pdf_nodes_Not_survived)

plt.figure(figsize=(9,6))
plt.plot(bin_edges_S[1:],pdf_nodes_survived)
plt.plot(bin_edges_S[1:],cdf_nodes_survived)

plt.plot(bin_edges_NS[1:],pdf_nodes_Not_survived)
plt.plot(bin_edges_NS[1:],cdf_nodes_Not_survived)


plt.ylabel('Density')
plt.xlabel('Auxillary Nodes')
plt.legend(['PDF of Auxillary Nodes Survived','CDF of Auxillary Nodes Survived','PDF of Auxillary Nodes Not Survived','CDF of Auxillary Nodes Not Survived'])
plt.show()
plt.figure(figsize=(8,5))
plt.plot(Survived["age"], np.zeros_like(Survived["age"]), '*', label = "Survived")
plt.plot(Not_Survived["age"], np.zeros_like(Not_Survived["age"]), '*', label = "Not Survived")
plt.title("scatter plot for Age")
plt.xlabel("age")
plt.legend()
plt.show()
sns.pairplot(df, hue = "status",vars = ["age", "year", "nodes"], height = 3)
plt.show()
print('Medians:')
print(np.median(Survived['nodes']))
print(np.median(np.append(Survived['nodes'],50)))
print(np.median(Not_Survived['nodes']))

print('\nQuantiles:')
print(np.percentile(Survived['nodes'],np.arange(0,100,25)))
print(np.percentile(Not_Survived['nodes'],np.arange(0,100,25)))

print('\n90th percentile:')
print(np.percentile(Survived['nodes'],90))
print(np.percentile(Not_Survived['nodes'],90))

print ('\nMedian Absolute Deviation')
print(robust.mad(Survived['nodes']))
print(robust.mad(Not_Survived['nodes']))
