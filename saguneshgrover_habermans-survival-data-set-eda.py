#importing all libraries 
import pandas as pd
import seaborn  as se
import numpy as np
import matplotlib.pyplot as plt

#reading the dataset
hb = pd.read_csv("../input/haberman/haberman.csv")
#hb
hb.shape
#it shows we have 306 rows and 4 columns
hb.columns
hb['survival_status'].value_counts();
se.FacetGrid(hb,hue="survival_status",size=5)\
    .map(se.distplot,"year")\
    .add_legend()
plt.show()
se.FacetGrid(hb,hue="survival_status",size=5)\
    .map(se.distplot,"Age")\
    .add_legend()
plt.show()
se.FacetGrid(hb,hue="survival_status",size=5)\
    .map(se.distplot,"positive_axillary_nodes")\
    .add_legend()
plt.show()
se.boxplot(x = 'survival_status',y = 'year',data = hb)
plt.show()
se.boxplot(x = 'survival_status',y = 'Age',data = hb)
plt.show()
se.boxplot(x = 'survival_status',y = 'positive_axillary_nodes',data = hb)
plt.show()
se.violinplot(x="survival_status", y="year", data=hb, size=8)
plt.show()
se.violinplot(x="survival_status", y="Age", data=hb, size=8)
plt.show()
se.violinplot(x="survival_status", y="positive_axillary_nodes", data=hb, size=8)
plt.show()
#pdf cdf of year

counts,bin_edges = np.histogram(hb['year'],bins = 30, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend()

counts,bin_edges = np.histogram(hb['year'],bins = 30, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


plt.xlabel('Year')
plt.grid()

plt.show()
#pdf cdf of positive_axillary_nodes

counts,bin_edges = np.histogram(hb['positive_axillary_nodes'],bins = 30, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend()

counts,bin_edges = np.histogram(hb['positive_axillary_nodes'],bins = 30, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.xlabel('positive_axillary_nodes')
plt.grid()

plt.show()
#pdf cdf of Age

counts,bin_edges = np.histogram(hb['Age'],bins = 30, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend()

counts,bin_edges = np.histogram(hb['Age'],bins = 30, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.xlabel('Age')
plt.grid()

plt.show()
se.set_style("darkgrid");
se.FacetGrid(hb,hue='survival_status',size=6)\
    .map(plt.scatter,"year","Age")\
    .add_legend();
plt.show()
se.set_style("darkgrid");
se.FacetGrid(hb,hue='survival_status',size=6)\
    .map(plt.scatter,"positive_axillary_nodes","Age")\
    .add_legend();
plt.show()
plt.close();
se.set_style("whitegrid");
se.pairplot(hb,hue="survival_status",size=3)
plt.show()

#hb is the name of the data frame
less_five = hb[hb['survival_status']==2]
more_five = hb[hb['survival_status']==1]
print(np.mean(more_five))
print(np.mean(less_five))
