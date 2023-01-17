import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

hb = pd.read_csv('../input/habermans-survival-data-set/haberman.csv',names= ['age','op_year','nodes','survived'])
print(hb.head())
#Data points and freatures
print(hb.shape)
#Column names in the dataset
print(hb.columns)
#modifying 'survived' feature from 1,2 to categories Yes or no 
hb['survived'] = hb['survived'].map({1:"yes", 2:"no"})
hb['survived'] = hb['survived'].astype('category')
print(hb.head())
#No of prople survived
hb['survived'].value_counts()
#PDF analysis of all the features(fig 1.1)
sns.FacetGrid(hb, hue="survived", height=5) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.show();
#fig(1.2)
sns.FacetGrid(hb, hue="survived", height=5) \
   .map(sns.distplot, "op_year") \
   .add_legend();
plt.show();
#fig(1.3)
sns.FacetGrid(hb, hue="survived", height=5) \
   .map(sns.distplot, "nodes") \
   .add_legend();
plt.show();
#cdf analysis of all features(fig2.1)
counts, bin_edges = np.histogram(hb['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)

#fig(2.2)
counts, bin_edges = np.histogram(hb['op_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)

#fig(2.3)
counts, bin_edges = np.histogram(hb['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)

#fig(3.1)
sns.boxplot(x ='survived',y = 'age',data = hb)
plt.show()
#fig(3.2)
sns.boxplot(x ='survived',y = 'op_year',data = hb)
plt.show()
#fig(3.3)
sns.boxplot(x ='survived',y = 'nodes',data = hb)
plt.show()
#fig(4.1)
sns.violinplot(x="survived", y="age", data=hb, size=8)
plt.show()
#fig(4.2)
sns.violinplot(x="survived", y="op_year", data=hb, size=8)
plt.show()
#fig(4.3)
sns.violinplot(x="survived", y="nodes", data=hb, size=8)
plt.show()
sns.set_style("whitegrid");
sns.pairplot(hb, hue="survived", vars = ["age", "op_year", "nodes"], size=3);
plt.show()
