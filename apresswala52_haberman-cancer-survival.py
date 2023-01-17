# Importing the required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
hsd = pd.read_csv('../input/haberman.csv') # hsd stands for Haberman Survival Dataset

# Info about the dataset
hsd.info()
# No. of datapoints and features
hsd.shape
# columns in the dataset
hsd.columns
# Datapoints for survival status
hsd['survival_status'].value_counts()
hsd.describe()
sns.FacetGrid(hsd, hue="survival_status", size=5) \
   .map(sns.distplot, "Age") \
   .add_legend();
plt.show();
sns.FacetGrid(hsd, hue="survival_status", size=5) \
   .map(sns.distplot, "year") \
   .add_legend();
plt.show();
sns.FacetGrid(hsd, hue="survival_status", size=5) \
   .map(sns.distplot, "positive_axillary_nodes") \
   .add_legend();
plt.show();
# PDF & CDF for Age
status_1 = hsd.loc[hsd["survival_status"] == 1];
status_2 = hsd.loc[hsd["survival_status"] == 2];

counts, bin_edges = np.histogram(status_1['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
#PDF & CDF for Operation Year
counts, bin_edges = np.histogram(status_1['year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
#PDF & CDF for Axillary Nodes
counts, bin_edges = np.histogram(status_1['positive_axillary_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
# Box Plots for Age
sns.boxplot(x='survival_status',y='Age', data=hsd)
plt.show()
# Box Plots for Axil_Nodes
sns.boxplot(x='survival_status',y='positive_axillary_nodes', data=hsd)
plt.show()
# Box Plots for Operation Year
sns.boxplot(x='survival_status',y='year', data=hsd)
plt.show()
#Violin Plot for axillary nodes
sns.violinplot(x='survival_status',y='positive_axillary_nodes', data=hsd, size=8)
plt.show()
#Violin Plot for Age
sns.violinplot(x='survival_status',y='Age', data=hsd, size=8)
plt.show()
#Violin Plot for Operation year
sns.violinplot(x='survival_status',y='year', data=hsd, size=8)
plt.show()
plt.close()
sns.set_style("whitegrid");
sns.pairplot(hsd, hue="survival_status", size=4);
plt.show()