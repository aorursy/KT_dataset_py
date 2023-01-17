import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns









#load csv file

#using head function to see a part of data

data=pd.read_csv('../input/haberman.csv')

data.head(10)
#name of columns

data.columns
#shape function to get number of rows and columns

data.shape
#describe() to get some more details such as mean ,standard deviation ,quantiles etc..

data.describe()
sns.FacetGrid(data,hue='Survival_Status',size=5).map(sns.distplot,"Age_Of_Patient").add_legend()

plt.show()
sns.FacetGrid(data,hue='Survival_Status',size=5).map(sns.distplot,"Operation_Year").add_legend()

plt.show()
sns.FacetGrid(data,hue='Survival_Status',size=5).map(sns.distplot,"Axil_nodes").add_legend()

plt.show()
counts,bin_edges=np.histogram(data['Survival_Status'],bins=10,density=True)

pdf=counts/(sum(counts))

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.show()
counts,bin_edges=np.histogram(data['Age_Of_Patient'],bins=10,density=True)

pdf=counts/(sum(counts))

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.show()
counts,bin_edges=np.histogram(data['Operation_Year'],bins=10,density=True)

pdf=counts/(sum(counts))

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.show()
counts,bin_edges=np.histogram(data['Axil_nodes'],bins=10,density=True)

pdf=counts/(sum(counts))

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.show()