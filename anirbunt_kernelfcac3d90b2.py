import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style

import os
print(os.curdir)
#print(os.__doc__)
print(os.listdir())

#pd.read_csv.__code__.co_varnames
df=pd.read_csv("../input/haberman.csv",header=None)
df.columns=['age','year','nodes','status']
print(df.info())
print(df.describe())
print(df)
temp=df
print(df['status'].value_counts())
df[df['status']==1].count()/306
df['nodes'][df["nodes"]<5].value_counts()
#sns.pairplot.__code__.co_varnames
sns.set()
style.use("ggplot")
sns.set_style("whitegrid");
sns.pairplot(data=df,hue='status',size=2.5)
plt.show()

sns.FacetGrid(df,hue='status').map(sns.distplot,"age").add_legend()
plt.show()
sns.FacetGrid(df,hue='status').map(sns.distplot,"year").add_legend()
plt.show()
sns.FacetGrid(df,hue='status').map(sns.distplot,"nodes").add_legend()
plt.show()
counts,bin_edges =np.histogram(df['age'],bins=10) #,density=True)
print("age counts {}\n".format(counts))
print("age bin_edges {}\n".format(bin_edges))
cumcounts=np.cumsum(counts)
print("age cumcounts {}\n".format(cumcounts))
densitycounts=counts/counts.sum()
print("age pdfcounts {}\n".format(densitycounts))
cumdensitycounts=np.cumsum(densitycounts)
print(cumdensitycounts)

countsyear,bin_years = np.histogram(df['year'],bins=10)
countsyear=countsyear/306
cumcountsyear=np.cumsum(countsyear)

countsnodes,bin_nodes = np.histogram(df['nodes'],bins=10)
countsnodes=countsnodes/306
cumcountsnodes=np.cumsum(countsnodes)

plt.subplot(1, 3, 1)
plt.plot(bin_edges[1:],densitycounts)
plt.plot(bin_edges[1:],cumdensitycounts)
plt.xlabel('Age')
plt.ylabel('cdf')

plt.subplot(1,3,2)
plt.plot(bin_years[1:],countsyear)
plt.plot(bin_years[1:],cumcountsyear)
plt.xlabel('Year')
plt.ylabel('Cdf')

plt.subplot(1,3,3)
plt.plot(bin_nodes[1:],countsnodes)
plt.plot(bin_nodes[1:],cumcountsnodes)
plt.xlabel('Nodes')
plt.ylabel('Cdf')
plt.title("\nTrends in entire data\n")
plt.show()

df=temp[temp['status']==1]
counts,bin_edges =np.histogram(df['age'],bins=10) #,density=True)
print("age counts {}\n".format(counts))
print("age bin_edges {}\n".format(bin_edges))

densitycounts=counts/counts.sum()
print("age pdfcounts {}\n".format(densitycounts))
cumdensitycounts=np.cumsum(densitycounts)
print(cumdensitycounts)


countsyear,bin_years = np.histogram(df['year'],bins=10)
countsyear=countsyear/countsyear.sum()
cumcountsyear=np.cumsum(countsyear)

countsnodes,bin_nodes = np.histogram(df['nodes'],bins=10)
countsnodes=countsnodes/countsnodes.sum()
cumcountsnodes=np.cumsum(countsnodes)



plt.subplot(1, 3, 1)
plt.plot(bin_edges[1:],densitycounts)
plt.plot(bin_edges[1:],cumdensitycounts)
plt.xlabel('Age')
plt.ylabel('cdf')

plt.subplot(1,3,2)
plt.plot(bin_years[1:],countsyear)
plt.plot(bin_years[1:],cumcountsyear)
plt.xlabel('Year')
plt.ylabel('Cdf')

plt.subplot(1,3,3)
plt.plot(bin_nodes[1:],countsnodes)
plt.plot(bin_nodes[1:],cumcountsnodes)
plt.xlabel('Nodes')

plt.ylabel('Cdf')
plt.title("\ntrends in patients who survived more than 5 years\n")
plt.show()


df=temp[temp['status']==2]
counts,bin_edges =np.histogram(df['age'],bins=10) #,density=True)
print("age counts {}\n".format(counts))
print("age bin_edges {}\n".format(bin_edges))

densitycounts=counts/counts.sum()
print("age pdfcounts {}\n".format(densitycounts))
cumdensitycounts=np.cumsum(densitycounts)
print(cumdensitycounts)

countsyear,bin_years = np.histogram(df['year'],bins=10)
countsyear=countsyear/countsyear.sum()
cumcountsyear=np.cumsum(countsyear)

countsnodes,bin_nodes = np.histogram(df['nodes'],bins=10)
countsnodes=countsnodes/countsnodes.sum()
cumcountsnodes=np.cumsum(countsnodes)



plt.subplot(1, 3, 1)
plt.plot(bin_edges[1:],densitycounts)
plt.plot(bin_edges[1:],cumdensitycounts)
plt.xlabel('Age')
plt.ylabel('cdf')

plt.subplot(1,3,2)
plt.plot(bin_years[1:],countsyear)
plt.plot(bin_years[1:],cumcountsyear)
plt.xlabel('Year')
plt.ylabel('Cdf')

plt.subplot(1,3,3)
plt.plot(bin_nodes[1:],countsnodes)
plt.plot(bin_nodes[1:],cumcountsnodes)
plt.xlabel('Nodes')

plt.ylabel('Cdf')
plt.title("\ntrends in patients who couldn't survived more than 5 years\n")
plt.show()

print("chart -1")

new =temp
new['bins'] = pd.cut(new['year'], 5)
print(new['bins'].value_counts())
print("survivals for>5 years")
new1=temp[temp['status']==1]
new1['bins'] = pd.cut(new1['year'], 5)
print(new1['bins'].value_counts())
print("survivals for <5 years")
new2=temp[temp['status']==2]
new2['bins'] = pd.cut(new2['year'], 5)
print(new2['bins'].value_counts())



print("chart -3")
new =temp
new['bins'] = pd.cut(new['age'], 5)
new[['bins', 'status']].groupby(['bins'], as_index=False).mean().sort_values(by='bins', ascending=True)


print("chart -4")
new =temp
new['bins'] = pd.cut(new['year'], 5)
print(new[['bins', 'nodes']].groupby(['bins'], as_index=False).median().sort_values(by='bins', ascending=True))
print(new[['bins', 'nodes']].groupby(['bins'], as_index=False).mean().sort_values(by='bins', ascending=True))
