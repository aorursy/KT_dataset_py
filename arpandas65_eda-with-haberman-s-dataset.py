#importing necessary Libraries 



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



#Loading Haberman Dataset



hman=pd.read_csv('../input/haberman/haberman.csv')



# Number of Data Points



print(hman.shape)
# Columns of the Dataset



print(hman.columns)
# Overview of the Data



print(hman.tail())
# modify the target column values to be meaningful and categorical



hman['Survival_status_after_5_years'] = hman['Survival_status_after_5_years'].map({1:"yes", 2:"no"})

hman['Survival_status_after_5_years'] = hman['Survival_status_after_5_years'].astype('category')



# printing top of modified data



print(hman.head())
# Number of Datapoints wrt 'Survival_status_afer_5_years'



hman['Survival_status_after_5_years'].value_counts()





# The Haberman Dataset is an imbalance Dataset with yes = 225 , no = 81



# High Level Statistics



print(hman.describe())
# plotting probability distribution wrt Patient's Age



plt.close()

sns.FacetGrid(hman,hue='Survival_status_after_5_years',size=5).map(sns.distplot,'Age').add_legend()

plt.show()

# plotting probability distribution wrt Patient's Year of Opearation



plt.close()

sns.FacetGrid(hman,hue='Survival_status_after_5_years',size=5).map(sns.distplot,'Operation_Year').add_legend()

plt.show()
# plotting probability distribution wrt Patient's Year of Opearation

plt.close()

sns.FacetGrid(hman,hue='Survival_status_after_5_years',size=6).map(sns.distplot,'axillary_lymph_nodes').add_legend()

plt.show()
# CDF wrt Patient's Age



label = ["pdf of survived", "cdf of survived", "pdf of not survived", "cdf of not survived"]

hman_survived=hman.loc[hman['Survival_status_after_5_years']=='yes']

hman_not_survived=hman.loc[hman['Survival_status_after_5_years']=='no']

counts,bin_edges = np.histogram(hman_survived['Age'],bins=15)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.figure(1)

#plt.subplot(211)

plt.plot(bin_edges[1:],pdf,bin_edges[1:],cdf)





#plt.subplot(212)

counts,bin_edges = np.histogram(hman_not_survived['Age'],bins=15)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.grid()

plt.plot(bin_edges[1:],pdf,bin_edges[1:],cdf)

plt.xlabel('Age')

plt.ylabel('number of patient')

plt.legend(label)





# CDF wrt Year of Operation



counts,bin_edges = np.histogram(hman_survived['Operation_Year'],bins=15)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,bin_edges[1:],cdf)







counts,bin_edges = np.histogram(hman_not_survived['Operation_Year'],bins=15)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.grid()

plt.plot(bin_edges[1:],pdf,bin_edges[1:],cdf)

plt.xlabel('Year of Operation')

plt.ylabel('number of patient')

plt.legend(label)
#CDF wrt axillary lymph nodes



counts,bin_edges = np.histogram(hman_survived['axillary_lymph_nodes'],bins=15)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.figure(1)

#plt.subplot(221)

plt.plot(bin_edges[1:],pdf,bin_edges[1:],cdf)





#plt.subplot(222)

counts,bin_edges = np.histogram(hman_not_survived['axillary_lymph_nodes'],bins=15)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.grid()

plt.plot(bin_edges[1:],pdf,bin_edges[1:],cdf)

plt.xlabel('axillary_lymph_nodes')

plt.ylabel('number of patient')

plt.legend(label)
# BoxPlot wrt patient's age



sns.boxplot(y='Age',x='Survival_status_after_5_years',data=hman)

plt.show()
# BoxPlot wrt year of operation



sns.boxplot(y='Operation_Year',x='Survival_status_after_5_years',data=hman)

plt.show()
# BoxPlot wrt axillary lymph nodes



sns.boxplot(y='axillary_lymph_nodes',x='Survival_status_after_5_years',data=hman)

plt.show()
# Violine plot wrt patient's age



sns.violinplot(y='Age',x='Survival_status_after_5_years',data=hman)

plt.show()
# Violin plot wrt year of operation



sns.violinplot(y='Operation_Year',x='Survival_status_after_5_years',data=hman)

plt.show()
# Violin plot wrt axillary lymph nodes



sns.violinplot(y='axillary_lymph_nodes',x='Survival_status_after_5_years',data=hman)

plt.show()
# we can draw 3C2 = 3 SCatter plots



sns.set_style('whitegrid')

sns.FacetGrid(hman,hue='Survival_status_after_5_years',size=6).map(plt.scatter,'Age','axillary_lymph_nodes').add_legend()

plt.show()

# Pair Plot 



plt.close()

sns.pairplot(hman,hue='Survival_status_after_5_years',size=5)

plt.show()
