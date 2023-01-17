#importing all important libraries and reading the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

survival_df = pd.read_csv('../input/haberman.csv',names=['age', 'year', 'axillary', 'survived'])
#view first few rows
survival_df.head(3)
#Computing Meta-data
print('The number of instances/records in the dataset: {0} and the Number of attributes: {1} \n'\
      .format(survival_df.shape[0],survival_df.shape[1]))
print('List of attributes:',survival_df.columns)
#Analysing Class distribution:
survival_df['survived'].value_counts()
#pairplots using seaborn
sns.set_style('whitegrid')
sns.pairplot(survival_df,hue='survived',vars=survival_df.columns[:-1],size=4,markers=['o','D'],palette='cubehelix')
plt.show()
#age
sns.FacetGrid(survival_df,hue='survived',size=6,palette='cubehelix') \
   .map(sns.distplot,'age') \
    .add_legend()
plt.show()    
#year
sns.FacetGrid(survival_df,hue='survived',palette='cubehelix',size=6)\
   .map(sns.distplot,'year')\
   .add_legend()
plt.show()    
#axillary nodes
sns.FacetGrid(survival_df,hue='survived',palette='cubehelix',size=6)\
    .map(sns.distplot,'axillary')\
    .add_legend()
plt.show()    
#age
survived = survival_df[survival_df['survived']==1]
not_survived = survival_df[survival_df['survived']==2]
fig,ax=plt.subplots(1,1,figsize=(14,8))
counts,bin_edges = np.histogram(survived['age'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='survived pdf')
plt.plot(bin_edges[1:],cdf,label='survived cdf')

counts,bin_edges = np.histogram(not_survived['age'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='not_survived pdf')
plt.plot(bin_edges[1:],cdf,label='not_survived cdf')

plt.legend()
plt.show()
#year
fig,ax=plt.subplots(1,1,figsize=(14,8))
counts,bin_edges = np.histogram(survived['year'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='survived pdf')
plt.plot(bin_edges[1:],cdf,label='survived cdf')

counts,bin_edges = np.histogram(not_survived['year'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='not_survived pdf')
plt.plot(bin_edges[1:],cdf,label='not_survived cdf')

plt.legend()
plt.show()
#Axillary Nodes
fig,ax=plt.subplots(1,1,figsize=(14,8))
counts,bin_edges = np.histogram(survived['axillary'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='survived pdf')
plt.plot(bin_edges[1:],cdf,label='survived cdf')

counts,bin_edges = np.histogram(not_survived['axillary'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='not_survived pdf')
plt.plot(bin_edges[1:],cdf,label='not_survived cdf')

plt.legend()
plt.show()
#Descriptive Statistics
survival_df.describe()
#age
sns.boxplot(data=survival_df,x='survived',y='age')
plt.show()
#year
sns.boxplot(data=survival_df,x='survived',y='year')
plt.show()
#no. of infected axillary nodes
sns.boxplot(data=survival_df,x='survived',y='axillary')
plt.show()
#age
sns.violinplot(data=survival_df,x='survived',y='age')
plt.show()
#year
sns.violinplot(data=survival_df,x='survived',y='year')
plt.show()
#axillary nodes
sns.violinplot(data=survival_df,x='survived',y='axillary')
plt.show()
'''Observations deduced from box plots holds true here and no new observations are found'''
