import pandas as pd

data = pd.read_csv('../input/habermans-survival-data-set/haberman.csv')
data.shape
data.head()
header_list=['age','op_year','axil_nodes','surv_status']

haberman_data = pd.read_csv('../input/habermans-survival-data-set/haberman.csv',names=header_list)
haberman_data.head()
haberman_data['surv_status'].value_counts()
import seaborn as sns

sns.set_style("whitegrid")



sns.FacetGrid(haberman_data,hue='surv_status',height=5).map(sns.distplot,'age').add_legend()
sns.FacetGrid(haberman_data,hue='surv_status',height=5).map(sns.distplot,'op_year').add_legend()
sns.FacetGrid(haberman_data,hue='surv_status',height=5).map(sns.distplot,'axil_nodes').add_legend()
survival_yes = haberman_data[haberman_data['surv_status']==1]

survival_no = haberman_data[haberman_data['surv_status']==2]
import numpy as np

import matplotlib.pyplot as plt

count, bin_edges = np.histogram(survival_no['age'], bins=10, density = True)

#count : the number of data points at that particular age value

#bin_edges :the seperation values of the X-axis (the feature under analysis)

#bins = the number of buckets of seperation

pdf = count/sum(count)

print(pdf)

# To get cdf, we want cumulative values of the count. In numpy, cumsum() does cumulative sum 

cdf = np.cumsum(pdf)

print(cdf)

count, bin_edges = np.histogram(survival_yes['age'], bins=10, density = True)

pdf2 = count/sum(count)

cdf2 = np.cumsum(pdf2)

plt.plot(bin_edges[1:],pdf,label='yes')

plt.plot(bin_edges[1:], cdf,label='yes')

plt.plot(bin_edges[1:],pdf2,label='no')

plt.plot(bin_edges[1:], cdf2,label='no')

plt.legend()

 #adding labels

plt.xlabel("AGE")

plt.ylabel("FREQUENCY")
count, bin_edges = np.histogram(survival_no['axil_nodes'], bins=10, density = True)

pdf = count/sum(count)

print(pdf)

cdf = np.cumsum(pdf)

print(cdf)

count, bin_edges = np.histogram(survival_yes['axil_nodes'], bins=10, density = True)

pdf2 = count/sum(count)

cdf2 = np.cumsum(pdf2)

plt.plot(bin_edges[1:],pdf,label='yes')

plt.plot(bin_edges[1:], cdf,label='yes')

plt.plot(bin_edges[1:],pdf2,label='no')

plt.plot(bin_edges[1:], cdf2,label='no')

plt.legend()

plt.xlabel("AXIL_NODES")

plt.ylabel("FREQUENCY")
sns.boxplot(x='surv_status',y='age', data=haberman_data)
sns.boxplot(x='surv_status',y='axil_nodes', data=haberman_data)
sns.boxplot(x='surv_status',y='op_year', data=haberman_data)
sns.violinplot(x='surv_status',y='age', data=haberman_data)

plt.show()
sns.violinplot(x='surv_status',y='op_year', data=haberman_data)

plt.show()
sns.violinplot(x='surv_status',y='axil_nodes', data=haberman_data)

plt.show()
sns.FacetGrid(haberman_data, hue="surv_status", height=8).map(plt.scatter, "age", "op_year").add_legend();
sns.FacetGrid(haberman_data, hue="surv_status", height=8).map(plt.scatter, "age", "axil_nodes").add_legend();
sns.FacetGrid(haberman_data, hue="surv_status", height=8).map(plt.scatter, "axil_nodes", "op_year").add_legend();
sns.pairplot(haberman_data, hue="surv_status", height=7)
g=sns.jointplot(x = 'op_year', y = 'age', data = haberman_data, kind = 'kde')
g=sns.jointplot(x = 'op_year', y = 'age', data = haberman_data, kind = 'hex')
#Hope you explored something intresting