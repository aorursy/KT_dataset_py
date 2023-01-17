import seaborn as sns

import pandas as pd

import numpy as np

from statsmodels import robust

import matplotlib.pyplot as plt 
haberman=pd.read_csv('../input/haberman.csv')
haberman.columns=['age','year','node','status']
haberman.columns

haberman.status=haberman.status.replace({1:"survived",2:"died"})
print("Number Of DataPoints : {} ".format(haberman.shape[0]))
print("Number Of Features : {}".format(haberman.shape[1]))
print("Number Of Classes : {}".format(len(haberman.status.value_counts())))
print(haberman.status.value_counts())
print("Independent Features Are : ")

i=0

for feature in haberman.columns:

    if i<3:

        print(str(i+1)+") "+feature)

    else:

        print("Dependent Feature Is : {}".format(feature))

    i+=1
patient_survived=haberman.loc[haberman.status=='survived']

patient_died=haberman.loc[haberman.status=='died']

plt.plot(patient_survived['age'],np.zeros_like(patient_survived['age']),'o',

         label='Survived')

plt.plot(patient_died['age'],np.zeros_like(patient_died['age']),'o',

         label='Died')

plt.xlabel("Age")

plt.title("1D Age Scatter Plot")

plt.legend()

plt.grid()

plt.show()
plt.plot(patient_survived['node'],np.zeros_like(patient_survived['node']),'o',label='Survived')

plt.plot(patient_died['node'],np.zeros_like(patient_died['node']),'o',label='Died')

plt.xlabel("Auxillary Nodes")

plt.title("1D Number of Nodes Scatter Plot")

plt.legend()

plt.grid()

plt.show()
sns.FacetGrid(haberman,hue='status',height=5).map(sns.distplot,"age").add_legend()

plt.grid()

plt.title("Age PDF")

plt.show()
sns.FacetGrid(haberman,hue='status',height=5).map(sns.distplot,"year").add_legend()

plt.grid()

plt.title("Year Of Operation PDF")

plt.show()
sns.FacetGrid(haberman,hue='status',height=6).map(sns.distplot,"node").add_legend()

plt.grid()

plt.title("Number Of Nodes PDF")

plt.show()
counts,bin_edges=np.histogram(patient_survived['age'],bins=10,density=True)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf-survived")

plt.plot(bin_edges[1:],cdf,label="cdf-survived")

counts,bin_edges=np.histogram(patient_died['age'],bins=10,density=True)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf-died")

plt.plot(bin_edges[1:],cdf,label="cdf-died")

plt.xlabel("Age")

plt.ylabel("Probabilities")

plt.legend()

plt.grid()

plt.title("Age PDF CDF")

plt.show()
counts,bin_edges=np.histogram(patient_survived['year'],bins=10,density=True)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf-survived")

plt.plot(bin_edges[1:],cdf,label="cdf-survived")

counts,bin_edges=np.histogram(patient_died['year'],bins=10,density=True)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf-died")

plt.plot(bin_edges[1:],cdf,label="cdf-died")

plt.xlabel("Year")

plt.ylabel("Probabilities")

plt.grid()

plt.title("Year Of Operation PDF CDF")

plt.legend()

plt.show()
counts,bin_edges=np.histogram(patient_survived['node'],bins=10,density=True)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf-survived")

plt.plot(bin_edges[1:],cdf,label="cdf-survived")

counts,bin_edges=np.histogram(patient_died['node'],bins=10,density=True)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf-died")

plt.plot(bin_edges[1:],cdf,label="cdf-died")

plt.legend()

plt.xlabel("Nodes")

plt.ylabel("Probabilities")

plt.grid()

plt.title("Number Of Nodes PDF CDF")

plt.show()
sns.boxplot(x='status',y='age',data=haberman)

plt.grid()

plt.title("Age Box Plot")

plt.show()
sns.boxplot(x='status',y='year',data=haberman)

plt.grid()

plt.title("Year Of Operation Box Plot")

plt.show()
sns.boxplot(x='status',y='node',data=haberman)

plt.grid()

plt.title("Number Of Nodes Box Plot")

plt.show()
sns.violinplot(x='status',y='age',data=haberman,size=10)

plt.grid()

plt.title("Age Violin Plot")

plt.show()
sns.violinplot(x='status',y='year',data=haberman,size=10)

plt.grid()

plt.title("Year Of Operation Violin Plot")

plt.show()
sns.violinplot(x='status',y='node',data=haberman,size=10)

plt.grid()

plt.title("Number Of Nodes Violin Plot")

plt.show()
plt.close()

sns.set_style("whitegrid")

sns.pairplot(haberman,hue="status",height=5)

plt.show()