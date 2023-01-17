import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



hbm=pd.read_csv("../input/haberman.csv")

hbm
hbm.status.dtype
#We shall label the class attribute 'status' in a readable format

hbm.status.replace([1,2],['survived 5 years or longer','died within 5 years'],inplace=True)

hbm
print(hbm.shape)

print(hbm.columns)
hbm.describe()
unique_years=list(hbm.year.unique())

unique_years.sort()

print('The list of unique years in which the operation was performed :' ,unique_years)

print('Number of unique Years:', len(unique_years))

unique_nodes=list(hbm.nodes.unique())

unique_nodes.sort()

print('The unique values of nodes:',unique_nodes)

print('The number of unique values of nodes are:',len(unique_nodes))

unique_age=list(hbm.age.unique())

unique_age.sort()

print('The list of unique age to which the operation was performed :' ,unique_age)

print('Number of unique age:', len(unique_age))
print(hbm.status.value_counts())

print('The percentage of people survived and died respectively is:',list(i*100 for i in list(hbm.status.value_counts(normalize=True))))

sns.countplot(x='status',data=hbm)

plt.show()
plt.figure(figsize=(15,5))

sns.set(style="ticks")

sns.countplot(x='age',data=hbm,hue='status')
sns.set(style="ticks")

sns.catplot(x='year',kind='count',data=hbm)

plt.show()

sns.catplot(x='year',kind='count',data=hbm,col='status')

plt.show()
sns.set(style="ticks")

sns.catplot(x='nodes',col='status',kind='count',data=hbm,height=6)

plt.show()
sns.set_style('whitegrid')

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for i,j in enumerate(list(hbm.columns[:-1])):

    sns.violinplot(x='status',y=j,data=hbm,ax=axes[i])

plt.show()
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for i,j in enumerate(list(hbm.columns[:-1])):

    sns.boxplot(x='status',y=j,data=hbm,ax=axes[i])

plt.show()
for i in hbm.columns[:-1]:

    sns.FacetGrid(hbm,hue='status',height=5).map(sns.distplot,i).add_legend()

plt.show()
for i in hbm.columns[:-1]:

    counts, bin_edges = np.histogram(hbm[i], bins=10,density = True)

    pdf = counts/(sum(counts))

    cdf = np.cumsum(pdf)

    print('PDF:',pdf) 

    print('CDF:',cdf)

    print('Bin edges:',bin_edges)

    plt.plot(bin_edges[1:],pdf)

    plt.plot(bin_edges[1:], cdf)

    plt.xlabel(i)

    plt.show()
for i in list(hbm.columns)[:-1]:

    print(i,':')

    print("Mean:")

    print(np.mean(hbm[i]))

    print("Median:")

    print(np.median(hbm[i]))

    print("Quantiles:")

    print(np.percentile(hbm[i],np.arange(0, 100, 25)))

    print("90th Percentiles:")

    print(np.percentile(hbm[i],90))

    from statsmodels import robust

    print ("Median Absolute Deviation")

    print(robust.mad(hbm[i]))
sns.FacetGrid(hbm,hue='status',height=8).map(plt.scatter,'nodes','age').add_legend()

plt.show()

sns.FacetGrid(hbm,hue='status',height=8).map(plt.scatter,'nodes','year').add_legend()

plt.show()
plt.close();

sns.set_style("whitegrid");

sns.pairplot(hbm,hue = "status",height = 4);

plt.show()
sns.jointplot(x='nodes',y='year',data=hbm,height=8)

plt.show()



sns.jointplot(x='age',y='nodes',data=hbm,kind='kde',height=8)

plt.show()