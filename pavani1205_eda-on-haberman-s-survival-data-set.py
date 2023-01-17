import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#reading the data

haberman=pd.read_csv('../input/haberman.csv')
#visualizing data

haberman.head(5)
#chnaging the column names

haberman.columns=['Age','Op_Year','axil_nodes','Surv_status']

haberman.head(5)
#finding the shape of data

haberman.shape
# finding out the number of data points for each class label

haberman['Surv_status'].value_counts()
sns.set_style('whitegrid')

sns.pairplot(haberman,hue='Surv_status',vars=['Age','Op_Year','axil_nodes'],size=3).add_legend()

plt.show()
#split the data

hab_surv=haberman.loc[haberman['Surv_status']==1]

hab_unsurv=haberman.loc[haberman['Surv_status']==2]
#pdf and cdf for feature age

counts,bin_edges=np.histogram(hab_surv['Age'],bins=10,density=True)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label='pdf_habsurv')

plt.plot(bin_edges[1:],cdf,label='cdf_habsurv')



counts,bin_edges=np.histogram(hab_unsurv['Age'],bins=10,density=True)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label='pdf_habunsurv')

plt.plot(bin_edges[1:],cdf,label='cdf_habunsurv')



plt.xlabel('Age')

plt.legend(loc='center left',bbox_to_anchor=(1,0.5))

plt.show()



#pdf and cdf of feature Op_Year

counts,bin_edges=np.histogram(hab_surv['Op_Year'],bins=10,density=True)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label='pdf_habsurv')

plt.plot(bin_edges[1:],cdf,label='cdf_habsurv')



counts,bin_edges=np.histogram(hab_unsurv['Age'],bins=10,density=True)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label='pdf_habunsurv')

plt.plot(bin_edges[1:],cdf,label='cdf_habunsurv')



plt.xlabel('Op_Year')

plt.legend(loc='center left',bbox_to_anchor=(1,0.5))

plt.show()



#pdf and cdf for feature axil_nodes

counts,bin_edges=np.histogram(hab_surv['axil_nodes'],bins=10,density=True)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label='pdf_habsurv')

plt.plot(bin_edges[1:],cdf,label='cdf_habsurv')



counts,bin_edges=np.histogram(hab_unsurv['Age'],bins=10,density=True)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label='pdf_habunsurv')

plt.plot(bin_edges[1:],cdf,label='cdf_habunsurv')



plt.xlabel('axil_nodes')

plt.legend(loc='center left',bbox_to_anchor=(1,0.5))

plt.show()











sns.boxplot(x='Surv_status',y='Age',data=haberman)

plt.show()



sns.boxplot(x='Surv_status',y='Op_Year',data=haberman)

plt.show()



sns.boxplot(x='Surv_status',y='axil_nodes',data=haberman)

plt.show()
sns.violinplot(x='Surv_status',y='Age',data=haberman)

plt.show()



sns.violinplot(x='Surv_status',y='Op_Year',data=haberman)

plt.show()



sns.violinplot(x='Surv_status',y='axil_nodes',data=haberman)

plt.show()