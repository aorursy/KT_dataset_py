import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid");
import os

print(os.listdir("../input"))
haberman = pd.read_csv("../input/haberman.csv",header=None, 
                       names=['age', 'year_of_operation', 'positive_axillary_nodes', 'survival_status'])
# (Q) how many data-points and features?
print (haberman.shape)
#(Q) How many data points for each class are present? 
haberman["survival_status"].value_counts()
# (Q) High Level Statistics
haberman.describe()
# modify the target column values to be meaningful as well as categorical
haberman['survival_status'] = haberman['survival_status'].map({1:"yes", 2:"no"})
haberman['survival_status'] = haberman['survival_status'].astype('category')
print(haberman.head())
print("# of rows: " + str(haberman.shape[0]))
print("# of columns: " + str(haberman.shape[1]))
print("Columns: " + ", ".join(haberman.columns))

print("Target variable distribution")
print(haberman.iloc[:,-1].value_counts())
print(haberman.iloc[:,-1].value_counts(normalize = True))
sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="survival_status", size=5) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.show();
sns.FacetGrid(haberman, hue="survival_status", size=5) \
   .map(sns.distplot, "year_of_operation") \
   .add_legend();
plt.show();
sns.FacetGrid(haberman, hue="survival_status", size=5) \
   .map(sns.distplot, "positive_axillary_nodes") \
   .add_legend();
plt.show();
counts, bin_edges = np.histogram(haberman['age'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.xlabel('age')

plt.show();
counts, bin_edges = np.histogram(haberman['positive_axillary_nodes'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.xlabel('positive_axillary_nodes')

plt.show();
counts, bin_edges = np.histogram(haberman['year_of_operation'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.xlabel('year_of_operation')

plt.show();
idx, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(x='survival_status',y='year_of_operation', data=haberman,ax=axes[0])
sns.boxplot(x='survival_status',y='age', data=haberman,ax=axes[1])
sns.boxplot(x='survival_status',y='positive_axillary_nodes', data=haberman,ax=axes[2])
plt.show()

idx, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.violinplot(x='survival_status',y='year_of_operation', data=haberman,ax=axes[0])
sns.violinplot(x='survival_status',y='age', data=haberman,ax=axes[1])
sns.violinplot(x='survival_status',y='positive_axillary_nodes', data=haberman,ax=axes[2])
plt.show()
sns.pairplot(haberman, hue='survival_status', size=4)
plt.show()