import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns

path='../input/haberman.csv'
hebarman=pd.read_csv(path)


#No. of data point and features
print(hebarman.shape)
#There are 305 observation in dataset
#adding columns names
hebarman.columns = ["age","year","axillary_node","survival_status"]
print(hebarman.columns)
hebarman['survival_status'].value_counts()

#Plotting plain scatter plot between axillary_node and survival
hebarman.plot(kind="scatter",x="axillary_node",y="survival_status")
plt.show()
sns.set_style("whitegrid")
sns.FacetGrid(hebarman,hue='survival_status',size=10)\
.map(plt.scatter,"age","axillary_node")\
.add_legend()
plt.show()
sns.set_style("whitegrid")
sns.pairplot(hebarman,hue='survival_status',size=3)
plt.show()
sns.FacetGrid(hebarman,hue='survival_status',size=5).map(sns.distplot,'year').add_legend()
plt.show()
sns.FacetGrid(hebarman,hue='survival_status',size=5).map(sns.distplot,'age').add_legend()
plt.show()
sns.FacetGrid(hebarman,hue='survival_status',size=5).map(sns.distplot,'axillary_node').add_legend()
plt.show()
survive = hebarman.loc[hebarman['survival_status']==1]
not_survive = hebarman.loc[hebarman['survival_status']==2]
counts, bin_edges = np.histogram(survive['axillary_node'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['Pdf for the patients who survive more than 5 years',
            'Cdf for the patients who survive more than 5 years'])



plt.show();
counts, bin_edges = np.histogram(not_survive['axillary_node'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['Pdf for the patients who died within 5 years',
            'Cdf for the patients who died within 5 years'])



plt.show();
print(survive.describe())
print(not_survive.describe())
sns.boxplot(x='survival_status',y='axillary_node', data=hebarman)
plt.show()
sns.violinplot(x='survival_status',y='axillary_node', data=hebarman,size=8)
plt.show()
