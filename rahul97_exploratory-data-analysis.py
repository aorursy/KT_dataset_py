import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
col = ['Patient_age', 'Year_of_operation', 'pos_axillary_nodes', 'status']

df = pd.read_csv('../input/haberman.csv', names = col)
df.head()
df.shape
df['status'].value_counts()
sns.lmplot(fit_reg = False, data = df, y = 'pos_axillary_nodes', x = 'Patient_age')
sns.pairplot(df, hue = 'status')
sns.FacetGrid(df, hue = "status", size = 5).map(sns.distplot, "Patient_age").add_legend()

plt.show()
print("Mean age of patients survived:", round(np.mean(df[df['status'] == 1]['Patient_age'])))

print("Mean age of patients not survived:", round(np.mean(df[df['status'] == 2]['Patient_age'])))
sns.FacetGrid(df, hue = "status", size = 5).map(sns.distplot, "pos_axillary_nodes").add_legend()

plt.show()
sns.FacetGrid(df, hue = "status", size = 5).map(sns.distplot, "Year_of_operation").add_legend()

plt.show()
sur = df[df['status'] == 1]

sur.describe()
not_sur = df[df['status'] == 2]

not_sur.describe()
sns.violinplot(x='status', y='pos_axillary_nodes', data=df)
from statsmodels import robust

print("\n Median Absolute Deviation")

print(robust.mad(sur['pos_axillary_nodes']))

print(robust.mad(not_sur['pos_axillary_nodes']))
counts, bin_edges = np.histogram(not_sur['pos_axillary_nodes'], bins=30, 

                                 density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)

counts, bin_edges = np.histogram(sur['pos_axillary_nodes'], bins=30, 

                                 density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)

plt.legend('sur')

plt.legend(['Not_sur_pdf', 'Not_sur_cdf','Sur_pdf', 'Sur_cdf'])

plt.xticks(np.linspace(0,50,12), rotation=-45)

plt.xlabel("pos_axillary_node")

plt.show()
counts, bin_edges = np.histogram(not_sur['Year_of_operation'], bins=30, 

                                 density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)

counts, bin_edges = np.histogram(sur['Year_of_operation'], bins=30, 

                                 density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)

plt.legend('sur')

plt.legend(['Not_sur_pdf', 'Not_sur_cdf','Sur_pdf', 'Sur_cdf'])



plt.show()
sns.violinplot(x='status', y='Patient_age', data=df)
sns.violinplot(x='status', y='Year_of_operation', data = df)
sns.jointplot(x= 'Patient_age',kind = 'kde', y='Year_of_operation', data = df)

plt.show()