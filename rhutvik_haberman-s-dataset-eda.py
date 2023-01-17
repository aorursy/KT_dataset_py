import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

sns.set(color_codes=True)
#reading dataframe
df = pd.read_csv('../input/haberman.csv')
#finding number of rows and columns
print(df.shape)
print(df.columns)
#setting dataframe column names
df.columns = ['age', 'operation_year', 'auxil_node', 'survival_status']
print(df.head())
print(df.info())
print(df['survival_status'].unique())
print(df['survival_status'].value_counts())
#mapping numeric values to categorical
df['survival_status'] = df['survival_status'].map({1:"YES",2:"NO"})
df['survival_status'] = df['survival_status'].astype('category')
print(df.describe())
#plotting PDF and histograms for age data
fg = sns.FacetGrid(df, hue='survival_status', size=5)
fg.map(sns.distplot, 'age').add_legend()
plt.title('Histogram and PDF for age - Haberman Dataset')
plt.show()
plt.close()
#plotting PDF and histograms for operation_year data
fg = sns.FacetGrid(df, hue='survival_status', size=5)
fg.map(sns.distplot, 'operation_year').add_legend()
plt.title('Histogram and PDF for opearation year - Haberman Dataset')
plt.show()
plt.close()
#plotting PDF and histograms for auxil_node data
fg = sns.FacetGrid(df, hue='survival_status', size=5)
fg.map(sns.distplot, 'auxil_node').add_legend()
plt.title('Histogram and PDF for number of positive auxilary node - Haberman Dataset')
plt.show()
plt.close()
#CDF for age data
counts, bin_edges = np.histogram(df['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
fig, ax = plt.subplots()
ax.plot(bin_edges[1:], pdf, label = 'pdf')
ax.plot(bin_edges[1:], cdf, label = 'cdf')
plt.xlabel('age')
plt.ylabel('probability')
plt.title('PDF and CDF for age data')
legend = ax.legend(fontsize='x-large')
plt.show()
plt.close()
#CDF for opeartion year data
counts, bin_edges = np.histogram(df['operation_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
fig, ax = plt.subplots()
ax.plot(bin_edges[1:],pdf)
ax.plot(bin_edges[1:], cdf)
plt.xlabel('operation year')
plt.ylabel('probability')
plt.title('PDF and CDF for operation year data')
legend = ax.legend(fontsize='x-large')
plt.show()
plt.close()
#CDF for positive auxilary node data
counts, bin_edges = np.histogram(df['auxil_node'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
fig, ax = plt.subplots()
ax.plot(bin_edges[1:],pdf)
ax.plot(bin_edges[1:], cdf)
plt.xlabel('auxilary nodes')
plt.ylabel('probability')
plt.title('PDF and CDF for number of positive auxilary data')
legend = ax.legend(fontsize='x-large')
plt.show()
plt.close()
#plotting boxplot for each column in dataset
fig, axes = plt.subplots(1, 3)
for index, feature in enumerate(list(df.columns)[:-1]):
    fg = sns.boxplot( x='survival_status', y = feature, data = df, ax = axes[index])
plt.show()
fig, axes = plt.subplots(1, 3)
for index, feature in enumerate(list(df.columns)[:-1]):
    fg = sns.violinplot( x='survival_status', y = feature, data = df, ax = axes[index])
plt.show()
#pairplotting dataframe to identify features which are useful in seperating survival status
fg = sns.pairplot(df, hue = 'survival_status', size = 3)
plt.subplots_adjust(top=0.9)
fg.fig.suptitle('pairplotting Haberman dataset features')
plt.show()
plt.close()
#contour plot
fg = sns.jointplot(x="operation_year", y="auxil_node", data=df, kind="kde")
plt.title('contour plot')
plt.show()
