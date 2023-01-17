import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

#load haberman.csv into a pandas dataframe
hm = pd.read_csv("../input/haberman.csv",header=None,names=["Age","YearOfOperation","AuxNodes","SurvivalStatus"])
# (Q) how many datapoints and features
print(hm.shape)
print(hm.columns)
hm["SurvivalStatus"].value_counts()
hm.describe()
sns.set_style("whitegrid");
sns.pairplot(hm, hue="SurvivalStatus",vars=["Age","AuxNodes","YearOfOperation"],size = 4)
plt.show();
sns.FacetGrid(hm, hue="SurvivalStatus",size=7)\
    .map(sns.distplot, "YearOfOperation")\
    .add_legend();
plt.show();
sns.FacetGrid(hm, hue = "SurvivalStatus", size=7)\
    .map(sns.distplot, "Age")\
    .add_legend();
plt.show();
sns.FacetGrid(hm, hue = "SurvivalStatus", size = 7)\
    .map(sns.distplot, "AuxNodes")\
    .add_legend()
ss_1 = hm.loc[hm["SurvivalStatus"] == 1];
ss_2 = hm.loc[hm["SurvivalStatus"] == 2];

counts, bin_edges = np.histogram(ss_1["Age"], bins=10, density = True)

pdf = counts/sum(counts)
print(pdf)
print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf, label = 'PDF-SurvivalStatus:1')
plt.plot(bin_edges[1:], cdf, label = 'CDF-SurvivalStatus:1')

counts, bin_edges = np.histogram(ss_2["Age"], bins=10, density = True)

pdf = counts/sum(counts)
print(pdf)
print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf, label = 'PDF-SurvivalStatus:2')
plt.plot(bin_edges[1:], cdf, label = 'CDF-SurvivalStatus:2')

plt.legend()
plt.xlabel("Age")
plt.show()
counts, bin_edges = np.histogram(ss_1["AuxNodes"], bins = 10, density = True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = 'PDF-SurvivalStatus:1')
plt.plot(bin_edges[1:], cdf, label = 'CDF-SurvivalStatus:1')

counts, bin_edges = np.histogram(ss_2["AuxNodes"], bins=10, density = True)

pdf = counts/sum(counts)
print(pdf)
print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf, label = 'PDF-SurvivalStatus:2')
plt.plot(bin_edges[1:], cdf, label = 'CDF-SurvivalStatus:2')

plt.legend()
plt.xlabel("AuxNodes")
plt.show()
count, bin_edges = np.histogram(ss_1["YearOfOperation"], bins = 10, density = True)
pdf = count/sum(count)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = 'PDF-SurvivalStatus:1')
plt.plot(bin_edges[1:], cdf, label = 'CDF-SurvivalStatus:1')

counts, bin_edges = np.histogram(ss_2["YearOfOperation"], bins=10, density = True)

pdf = counts/sum(counts)
print(pdf)
print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf, label = 'PDF-SurvivalStatus:2')
plt.plot(bin_edges[1:], cdf, label = 'CDF-SurvivalStatus:2')

plt.legend()
plt.xlabel("YearOfOperation")
plt.show()
sns.boxplot(x="SurvivalStatus", y = "Age", data = hm)
plt.show()
sns.violinplot(x="SurvivalStatus", y="Age", data = hm, size = 5)
plt.show()
sns.boxplot(x="SurvivalStatus", y="AuxNodes", data = hm)
plt.show()
sns.violinplot(x="SurvivalStatus", y="AuxNodes", data = hm, size = 5)
plt.show()
sns.boxplot(x="SurvivalStatus", y="YearOfOperation", data = hm)
plt.show()
sns.violinplot(x="SurvivalStatus", y="YearOfOperation", data = hm, size = 5)
plt.show()