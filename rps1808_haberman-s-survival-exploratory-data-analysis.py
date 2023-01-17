#1.1
# Setting up the environment and storing data in a dataframe

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#haberman_df=pd.read_csv('haberman.csv')   # storing the given details into a dataframe
col = ['Age', 'Operation_year', 'Axil_nodes', 'Surv_status']
haberman = pd.read_csv('../input/haberman.csv', names = col)
print(haberman.info())
print("Distribution of records:\n",haberman['Surv_status'].value_counts())
print("Distribution of records in %\n", haberman['Surv_status'].value_counts(normalize=True)*100)

sns.pairplot(haberman,hue='Surv_status',size=4)
haberman_serv_gt5yr=haberman.loc[haberman['Surv_status']==1]
haberman_not_serv=haberman.loc[haberman['Surv_status']==2]

counts, bin_edges= np.histogram(haberman_serv_gt5yr['Age'],bins=10,density=True)
pdf=counts/(sum(counts))
cdf=np.cumsum(pdf)
print(pdf)
print(cdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

counts, bin_edges=np.histogram(haberman_serv_gt5yr['Operation_year'],bins=10,density=True)
pdf=counts/(sum(counts))
cdf=np.cumsum(pdf)
print(pdf)
print(cdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

counts, bin_edges=np.histogram(haberman_serv_gt5yr['Axil_nodes'],bins=10,density=True)
pdf=counts/(sum(counts))
cdf=np.cumsum(pdf)
print(pdf)
print(cdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
counts, bin_edges= np.histogram(haberman_not_serv['Age'],bins=10,density=True)
pdf=counts/(sum(counts))
cdf=np.cumsum(pdf)
print(pdf)
print(cdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

counts, bin_edges=np.histogram(haberman_not_serv['Operation_year'],bins=10,density=True)
pdf=counts/(sum(counts))
cdf=np.cumsum(pdf)
print(pdf)
print(cdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

counts, bin_edges=np.histogram(haberman_not_serv['Axil_nodes'],bins=10,density=True)
pdf=counts/(sum(counts))
cdf=np.cumsum(pdf)
print(pdf)
print(cdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
print("Patients who have servived 5 years or more\n\n",haberman_serv_gt5yr.describe())
print("\n\n patients who died within 5 years \n\n",haberman_not_serv.describe())
sns.boxplot(x='Surv_status',y='Age',data=haberman)
sns.boxplot(x='Surv_status',y='Operation_year',data=haberman)
sns.boxplot(x='Surv_status',y='Axil_nodes',data=haberman)
sns.violinplot(x='Surv_status',y='Age',data=haberman)
sns.violinplot(x='Surv_status',y='Operation_year',data=haberman)
sns.violinplot(x='Surv_status',y='Axil_nodes',data=haberman)