import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

print('Libraries Imported')
import os
print(os.listdir('../input'))
#dataSetURL = '/resources/data/AppliedAI/EDA_ch8/haberman.csv'

labels = ['age', 'operation_year', 'axil_nodes', 'survived_status']

df = pd.read_csv('../input/haberman.csv', names = labels)
df.head()
df.shape
df.columns
df.describe()
df.info()
df['survived_status'] = df['survived_status'].map({1:'survived', 2:'dead'})
df.tail()
df['survived_status'].value_counts()
plt.scatter(df['age'],df['operation_year'], c = 'g')
plt.xlabel('Age')
plt.ylabel('Operation year')
plt.title('Operation year vs Age')
plt.show()
plt.scatter(df['age'],df['axil_nodes'], color = 'g')
plt.xlabel('Age')
plt.ylabel('Axil Nodes')
plt.title('Axil_nodes vs Age')
plt.show()
plt.scatter(df['axil_nodes'], df['operation_year'], c = 'g')
plt.xlabel('Axil Nodes')
plt.ylabel('Operation year')
plt.title('Operation year vs Axil Nodes')
plt.show()
plt.close();
sns.set_style('whitegrid');
sns.pairplot(df, hue = 'survived_status', size = 4)
plt.show()
sns.set_style('whitegrid');
sns.FacetGrid(df, hue = 'survived_status', size = 6)\
   .map(plt.scatter, 'age', 'axil_nodes')\
   .add_legend();
plt.show();
sns.set_style('whitegrid');
sns.FacetGrid(df, hue='survived_status', size = 7) \
    .map(plt.scatter, 'operation_year', 'axil_nodes') \
    .add_legend();
plt.show()
sns.FacetGrid(df, hue='survived_status', size = 5) \
    .map(sns.distplot, 'axil_nodes') \
    .add_legend();
plt.show();
sns.FacetGrid(df, hue='survived_status', size = 5) \
    .map(sns.distplot, 'age') \
    .add_legend();
plt.show();
sns.FacetGrid(df, hue='survived_status', size = 5) \
    .map(sns.distplot, 'operation_year') \
    .add_legend();
plt.show();
counts, bin_edges = np.histogram(df['axil_nodes'], bins=20, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.xlabel('Axil_nodes')
plt.show()
sns.boxplot(x='survived_status', y = 'axil_nodes', data=df)
plt.show()
sns.violinplot(x='survived_status', y='axil_nodes', data = df, size = 9)
plt.show()