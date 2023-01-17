import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
data=pd.read_csv("/kaggle/input/habermans-survival-data-set/haberman.csv")
data.head()
data.tail()
data.describe()
# renaming the columns

data=data.rename(columns={'30':'age','64':'op_year','1':'axill_nodes','1.1':'survival_status'})
data.shape
# data points per class
data.survival_status.value_counts()
# pdf of survival status w.r.t age
sns.FacetGrid(data,hue='survival_status',height=4).map(sns.distplot,'age').add_legend()
# pdf of survival status w.r.t. operation year
sns.FacetGrid(data,hue='survival_status',size=4).map(sns.distplot,'op_year').add_legend()
#pdf of survival status w.r.t axilliary lymph nodes
sns.FacetGrid(data,hue='survival_status',height=4).map(sns.distplot,'axill_nodes').add_legend()
data_not_survived=data.loc[data['survival_status']==2]
counts,bin_edges=np.histogram(data_not_survived['axill_nodes'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend('survival status')
plt.legend(['not_survived_pdf','not_survived_cdf'])
plt.xlabel('age')
plt.show()
data_surv=data.loc[data['survival_status']==1]
counts,bin_edges=np.histogram(data_surv['axill_nodes'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend('survival status')
plt.legend(['survived_pdf','survived_cdf'])
plt.xlabel('age')
plt.show()
counts,bin_edges=np.histogram(data_not_survived['axill_nodes'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

counts,bin_edges=np.histogram(data_surv['axill_nodes'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend('survival status')
plt.legend(['not_survived_pdf','not_survived_cdf','survived_pdf','survived_cdf'])
plt.xlabel('age')
plt.show()
sns.boxplot(x='survival_status',y='axill_nodes',data=data).set_title('boxplot of axilliary node and survival status')
plt.show()
sns.violinplot(x='survival_status',y='axill_nodes',data=data,height=5)
plt.show()
# 1-d scatter plot

sur=data.loc[data['survival_status']==1]
not_sur=data.loc[data['survival_status']==2]
plt.plot(sur['age'],np.zeros_like(sur['age']),'o',label='survival status 1')
plt.plot(not_sur['age'],np.zeros_like(not_sur['age']),'o',label='survival status 2')
plt.xlabel('age')
plt.title('1-D scatter plot for age')
plt.legend()
plt.show()
#2d scatter plot

# age and operation year
sns.FacetGrid(data,hue='survival_status',height=5).map(plt.scatter,'age','op_year').add_legend()
plt.title('scatter plot of age and operation year')
plt.show()
#age and axilliary lymph node

sns.FacetGrid(data,hue='survival_status',height=5).map(plt.scatter,'age','axill_nodes').add_legend()
plt.title('scatter plot for age and axilliary lymph nodes')
plt.show()
sns.pairplot(data,hue='survival_status',vars=['age','op_year','axill_nodes'],height=4)
plt.suptitle('pair plot of age, operation year and axilliary lymph nodes')
plt.show()