# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline
data = pd.read_csv(r'/kaggle/input/habermans-survival-data-set/haberman.csv')
data = data.rename(columns = {"30" : "age", "64" : "operation_year", "1" : "lymph_node", "1.1" : "survival_status"})
data.head()
data.tail()
data.info()
data.describe()
data['survival_status'].value_counts()
data['lymph_node'].value_counts()
sns.set_style('whitegrid')

sns.FacetGrid(data,hue='survival_status',height=8).map(plt.scatter,'survival_status','age')

plt.show()
sns.FacetGrid(data,hue='survival_status',height=8).map(sns.distplot,'age').add_legend()
sns.FacetGrid(data,hue='survival_status',height=8).map(sns.distplot,'operation_year').add_legend()
sns.FacetGrid(data,hue='survival_status',height=8).map(sns.distplot,'lymph_node').add_legend()
counts,bin_edges = np.histogram(data['age'],bins=10,density=True)

pdf = counts/sum(counts)

print(pdf)

print(bin_edges)



cdf = np.cumsum(pdf)  # cumsum = cumilative sum

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)
counts,bin_edges = np.histogram(data['lymph_node'],bins=10,density=True)

pdf = counts/sum(counts)

print(pdf)

print(bin_edges)



cdf = np.cumsum(pdf)  # cumsum = cumilative sum

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)
counts,bin_edges = np.histogram(data['operation_year'],bins=10,density=True)

pdf = counts/sum(counts)

print(pdf)

print(bin_edges)



cdf = np.cumsum(pdf)  # cumsum = cumilative sum

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)
sns.boxplot(y=data['operation_year'],x=data['survival_status'])

plt.show()
sns.violinplot(y=data['age'],x=data['survival_status'])

plt.show()
sns.jointplot(x='age',y='operation_year',data=data,kind='kde')

plt.show()
sns.pairplot(data,hue='survival_status',height=3)