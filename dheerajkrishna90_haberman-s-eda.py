import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
col =['Age','Year-of-operation','Pos-axillary-nodes','Status']

df = pd.read_csv('../input/haberman.csv',names = col)
df.columns
df.head()
#Number of data points and features

print(df.shape)
#Number of classes and points per class

df['Status'].value_counts()



#It is an imbalanced dataset
sns.set_style('whitegrid');

sns.FacetGrid(df, hue = 'Status',size = 4).map(plt.scatter,'Age', 'Pos-axillary-nodes').add_legend();

plt.show();
print("Mean age of patients who survived:",(round(np.mean(df[df['Status'] == 1]['Age']))))

print("Mean age of patients who didnot survive:",(round(np.mean(df[df['Status']==2]['Age']))))
sns.pairplot(data = df, hue = 'Status', size = 3)
sns.FacetGrid(df, hue = 'Status', size = 5).map(sns.distplot, "Age").add_legend();

plt.show();
sns.FacetGrid(df, hue = 'Status', size = 5).map(sns.distplot, "Year-of-operation").add_legend();

plt.show();
sns.FacetGrid(df, hue = 'Status', size = 5).map(sns.distplot, "Pos-axillary-nodes").add_legend();

plt.show();
sur = df[df['Status'] == 1]

not_sur = df[df['Status'] == 2]
counts,bin_edges = np.histogram(not_sur['Pos-axillary-nodes'],bins = 30, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.legend()



counts,bin_edges = np.histogram(sur['Pos-axillary-nodes'],bins = 30, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)



plt.legend(['sur_pdf','not_sur_pdf','sur_cdf','not_sur_cdf'])

plt.xlabel('Pos-axillary-nodes')

plt.grid()
sns.boxplot(x = 'Status', y = 'Pos-axillary-nodes' , data = df)

plt.show()
sns.violinplot(x = 'Status', y = 'Pos-axillary-nodes' , data = df)

plt.show()
print("\n90th Percentiles:")

print(np.percentile(sur["Pos-axillary-nodes"],90))

print(np.percentile(not_sur["Pos-axillary-nodes"],90))