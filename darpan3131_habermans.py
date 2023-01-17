import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#read csv file and assign it to pandas Dataframe

survival_df=pd.read_csv('../input/haberman.csv')
survival_df.head()
#Have a look at total rows in data along with datatype of features and number of non-null/null points in each coloumn.

survival_df.info()
#Find number of rows and coloumns.

survival_df.shape
#FInd name of all features

survival_df.columns
#High level statistics of given data

survival_df.describe()
sur = survival_df.loc[survival_df.status==1]  #dataframe of patients who survived

sur_no = survival_df.loc[survival_df.status==2]#dataframe of patients who do not survived after five years.

print(sur.describe())

print(sur_no.describe())
#Check if given data is balanced or imbalanced.

ax=sns.countplot(survival_df.status)

total=float(len(survival_df))

for p in ax.patches:

    height=p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height - 50,

            'count:'+'{:1}'.format(height)+'\npercentage:{:1.1f}'.format((height*100)/total),

            ha="center")

#survival_df.status.value_counts(normalize=True)
sns.set_style('whitegrid')

#df = survival_df[['age','year','nodes']]

sns.pairplot(survival_df,hue='status')
#fig,axes = plt.subplots(figsize=(18,4),nrows=1,ncols=3)

#plt.tight_layout()

#sns.distplot(survival_df.age,ax=axes[0])

#How to plot 3 distplot on same row using Facetgrid?(Doubt)

sns.set_style('whitegrid')

g0 = sns.FacetGrid(survival_df,hue='status',size=4).map(sns.distplot,'age',).add_legend()

g1 = sns.FacetGrid(survival_df,hue='status',size=4).map(sns.distplot,'year').add_legend()

g2 = sns.FacetGrid(survival_df,hue='status',size=4).map(sns.distplot,'nodes').add_legend()

plt.show()
survival_df.loc[(survival_df.nodes>=0) & (survival_df.nodes<=4)].status.value_counts(normalize=True)
plt.figure(figsize=(20,5))

label = ["pdf of survived", "cdf of survived", "pdf of not survived", "cdf of not survived"]

for idx,coloumn  in enumerate(list(survival_df.columns[:-1])):

    plt.subplot(1,3,idx+1)

    counts,bin_edges = np.histogram(sur[coloumn],bins=10,density=True)

    pdf=counts/sum(counts)

    cdf=np.cumsum(pdf)

    plt.plot(bin_edges[1:],pdf)

    plt.plot(bin_edges[1:],cdf)

    counts,bin_edges = np.histogram(sur_no[coloumn],bins=10,density=True)

    pdf=counts/sum(counts)

    cdf=np.cumsum(pdf)

    plt.plot(bin_edges[1:],pdf)

    plt.plot(bin_edges[1:],cdf)

    plt.title("PDF & CDF of feature: "+coloumn)

    plt.xlabel(coloumn)

    plt.legend(label)

    
fig,axes = plt.subplots(1,3,figsize=(15,5))

for idx,feature in enumerate(list(survival_df.columns)[:-1]):

    sns.boxplot(x='status',y=feature,data=survival_df,ax=axes[idx],hue='status').set_title('survival status vs '+feature)

plt.show()
fig,axes = plt.subplots(1,3,figsize=(15,5))

for idx,feature in enumerate(list(survival_df.columns)[:-1]):

    sns.violinplot(x='status',y=feature,data=survival_df,ax=axes[idx],hue='status').set_title('survival status vs '+feature)

    axes[idx].legend(loc='lower center')

#plt.legend(loc='best')    

plt.show()