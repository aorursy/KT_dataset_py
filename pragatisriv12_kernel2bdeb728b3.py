import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df=dataset = pd.read_csv("../input/haberman.csv",header=None,
        names=['age', 'year', 'nodes', 'status'])
print(df.shape)
print(df.columns)
df
df["status"].value_counts()
sns.set_style("whitegrid")
sns.FacetGrid(df,hue="status",height=3).map(plt.scatter,"age","year").add_legend()
plt.show()
plt.close()
sns.set_style("whitegrid")
sns.pairplot(df,hue="status",vars=[df.columns[0],df.columns[1],df.columns[2]],height=3)
plt.show()
df.describe()

#sinlge attribute analysis
for i,attrbt in enumerate(list(df.columns)[:-1]):
    sns.FacetGrid(df,hue="status",height=4).map(sns.distplot,attrbt).add_legend()
    plt.show()

before5=df.loc[df["status"]==2]
after5=df.loc[df["status"]==1]
counts,bin_edges=np.histogram(before5["nodes"],bins=10,density=True)
pdf=counts/(sum(counts))
print("PDF")
print(pdf)
print("BIN EDGES")
print(bin_edges)
cdf=np.cumsum(pdf)
print("CDF")
print(cdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.xlabel("Nodes")
counts,bin_edges=np.histogram(after5["nodes"],bins=10,density=True)
pdf=counts/(sum(counts))
print("PDF")
print(pdf)
print("BIN EDGES")
print(bin_edges)
cdf=np.cumsum(pdf)
print("CDF")
print(cdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.xlabel("Nodes")
print(np.percentile(after5['nodes'],(25,50,75)))
print(np.percentile(before5['nodes'],(25,50,75)))
fig,axes=plt.subplots(1,3,figsize=(15, 5))
for i,atrbt in enumerate(list(df.columns)[:-1]):
    sns.boxplot(x='status',y=atrbt,data=df,ax=axes[i])
plt.show()
print(np.percentile(after5['nodes'],(25,50,75)))
print(np.percentile(before5['nodes'],(25,50,75)))
fig,axes=plt.subplots(1,3,figsize=(15, 5))
for i,atrbt in enumerate(list(df.columns)[:-1]):
    sns.violinplot(x='status',y=atrbt,data=df,ax=axes[i])
plt.show()
print(np.percentile(after5['year'],(25,50,75)))
print(np.percentile(before5['year'],(25,50,75)))
print(np.percentile(after5['age'],(25,50,75)))
print(np.percentile(before5['age'],(25,50,75)))