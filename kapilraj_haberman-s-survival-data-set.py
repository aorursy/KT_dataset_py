

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/haberman.csv',names=['Age','OpYear','axilNodes', 'Survival_status'])

data.head(n=20).T
data['Survival_status'] = data['Survival_status'].map({1: 0 , 2 : 1})
data.head(n=20).T
data.info()
data.describe()
data.shape
data['Survival_status'].value_counts().unique()
data['Survival_status'].value_counts(normalize = True)
np.sum(data.isna())
#what is the mean age?

data['Age'].mean()
data['OpYear'][data['OpYear'].value_counts().max()]
import seaborn as sns
sns.pairplot(data=data, hue='Survival_status')
sns.FacetGrid(data, hue="Survival_status", height = 5).map(sns.distplot, 'Age').add_legend()
sns.FacetGrid(data, hue="Survival_status", height = 5).map(sns.distplot, 'OpYear').add_legend()
sns.FacetGrid(data, hue="Survival_status", height = 5).map(sns.distplot, 'axilNodes').add_legend()
plt.figure(figsize=(15,10))

plt.figure(1)

plt.subplot(211)

counts, bin_edges = np.histogram(data['axilNodes'], bins = 10 , density =True)

pdf = counts / sum(counts)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)

plt.plot(bin_edges[1:], cdf)

plt.xlabel('axilNodes')

plt.grid()



plt.subplot(212)

counts, bin_edges = np.histogram(data['Age'], bins = 10 , density =True)

pdf = counts / sum(counts)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)

plt.plot(bin_edges[1:], cdf)

plt.xlabel('Age')

plt.grid()

sns.boxplot(x= data['Survival_status'], y = data['Age'] )

sns.boxplot(x= data['Survival_status'], y = data['OpYear'])

sns.boxplot(x= data['Survival_status'], y = data['axilNodes'])

sns.violinplot(x= data['Survival_status'], y = data['Age'])

sns.violinplot(x= data['Survival_status'], y = data['OpYear'])

sns.violinplot(x= data['Survival_status'], y = data['axilNodes'])
