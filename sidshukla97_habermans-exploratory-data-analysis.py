# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

#for visualization
df = pd.read_csv('/kaggle/input/haberman/haberman.csv')
df.head()

#top-most values of the dataset
df.shape

#our dataset have 306 rows and 4 columns
df.columns

#columns of the dataset
df['status'].value_counts()
df.describe()
df.info()
sns.set_style("darkgrid");

sns.FacetGrid(df, hue='status', size=6).map(plt.scatter, "year", "nodes").add_legend();

plt.show();
sns.set_style('darkgrid');

sns.FacetGrid(df, hue='status' , size=6).map(plt.scatter, 'nodes', 'age').add_legend();

plt.show()
sns.FacetGrid(df, hue='status', size=5).map(sns.distplot,"year").add_legend()

plt.show()

#points are overlapping as we can see
sns.FacetGrid(df, hue='status', size=5).map(sns.distplot, 'age').add_legend()

plt.show()
sns.FacetGrid(df, hue='status', size=5).map(sns.distplot, "nodes").add_legend()

plt.show()
sns.boxplot(x='status', y='year', data=df)

plt.show()
sns.boxplot(x='status', y='age', data=df)

plt.show()
sns.boxplot(x='status', y='nodes', data=df)

plt.show()
#people most people who survived have zero positive axillary nodes
sns.violinplot(x='status' , y='nodes', data=df, size=8)

plt.show()
sns.violinplot(x='status', y='age', data=df, size=8)

plt.show()
sns.violinplot(x='status', y='year', data=df, size=8)

plt.show()
import plotly.express as px

fig = px.scatter_3d(df, x='age', y='nodes', z='year', color='status')

fig.show()
#pdf cdf of year

counts, bin_edges = np.histogram(df['year'], bins=30 , density=True)

pdf = counts/sum(counts)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)

plt.plot(bin_edges[1:], cdf)

plt.legend()



counts, bin_edges = np.histogram(df['year'], bins=30, density=True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)

plt.plot(bin_edges[1:], cdf)



plt.xlabel('Year')

plt.grid()

plt.show()
#pdf cdf of positive_axillary_nodes



counts,bin_edges = np.histogram(df['nodes'],bins = 30, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.legend()



counts,bin_edges = np.histogram(df['nodes'],bins = 30, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)



plt.xlabel('positive_axillary_nodes')

plt.grid()



plt.show()
#pdf cdf of Age



counts,bin_edges = np.histogram(df['age'],bins = 30, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.legend()



counts,bin_edges = np.histogram(df['age'],bins = 30, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)



plt.xlabel('Age')

plt.grid()



plt.show()
# pairwise scatter plot: Pair-Plot



plt.close();

sns.set_style("darkgrid");

sns.pairplot(df, hue='status', size=3)

plt.show()
survived_patients = df[df['status']==1]

not_survived = df[df['status']==2]

print(np.mean(survived_patients))
print(np.mean(not_survived))
sns.jointplot(x='age', y='nodes', data=df, kind='kde', heights=6)

plt.show()