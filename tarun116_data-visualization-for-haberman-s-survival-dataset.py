import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
haberman = pd.read_csv('../input/haberman.csv',names = ['Age','OperationYear','Axil_Nodes','Survival_Status'])
print(haberman.shape)
haberman['Survival_Status'].value_counts()
haberman.info()
haberman.Survival_Status.unique()
haberman['Survival_Status'] = haberman['Survival_Status'].map({1:True, 2:False})
#haberman['Survival_Status'] = haberman['Survival_Status'].astype('category')
haberman.head(5)
haberman.info()
haberman.describe()
haberman.plot(kind='scatter', x='Age', y='Axil_Nodes') ;
plt.show()
sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="Survival_Status", size=4) \
   .map(plt.scatter, "Age", "Axil_Nodes") \
   .add_legend();
plt.show();
plt.close();
sns.set_style("whitegrid");
sns.pairplot(haberman, hue="Survival_Status", size=3);
plt.show()
haberman_1 = haberman.loc[haberman["Survival_Status"] == True]
haberman_2 = haberman.loc[haberman["Survival_Status"] == False];

plt.plot(haberman_1["Age"], np.zeros_like(haberman_1["Age"]), 'o')
plt.plot(haberman_2["Age"], np.zeros_like(haberman_2["Age"]), 'o')

sns.FacetGrid(haberman, hue="Survival_Status", size=5) \
   .map(sns.distplot, "Age") \
   .add_legend();
plt.show();
sns.FacetGrid(haberman, hue="Survival_Status", size=5) \
   .map(sns.distplot, "OperationYear") \
   .add_legend();
plt.show();
sns.FacetGrid(haberman, hue="Survival_Status", size=5) \
   .map(sns.distplot, "Axil_Nodes") \
   .add_legend();
plt.show();
counts, bin_edges = np.histogram(haberman_1['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(haberman_1['Age'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);

plt.show();
counts, bin_edges = np.histogram(haberman_1['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show()
counts, bin_edges = np.histogram(haberman_1['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


# virginica
counts, bin_edges = np.histogram(haberman_2['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();
print("Means:")
print(np.mean(haberman_1['Age']))
print(np.mean(haberman_2['Age']))

print("\nStd-dev:");
print(np.std(haberman_1['Age']))
print(np.std(haberman_2['Age']))

print("\nMedians:")
print(np.median(haberman_1['Age']))
print(np.median(haberman_2['Age']))


print("\nQuantiles:")
print(np.percentile(haberman_1['Age'],np.arange(0, 100, 25)))
print(np.percentile(haberman_2['Age'],np.arange(0, 100, 25)))

print("\n90th Percentiles:")
print(np.percentile(haberman_1['Age'],90))
print(np.percentile(haberman_2['Age'],90))

from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(haberman_1['Age']))
print(robust.mad(haberman_2['Age']))

sns.boxplot(x='Survival_Status', y='Age', data=haberman)
plt.show()
sns.boxplot(x='Survival_Status',y='OperationYear', data=haberman)
plt.show()
sns.boxplot(x='Survival_Status',y='Axil_Nodes', data=haberman)
plt.show()
sns.violinplot(x="Survival_Status", y="Age", data=haberman, size=8)
plt.show()
sns.violinplot(x="Survival_Status", y="OperationYear", data=haberman, size=8)
plt.show()
sns.violinplot(x="Survival_Status", y="Axil_Nodes", data=haberman, size=8)
plt.show()
sns.jointplot(x="Age", y="Axil_Nodes", data=haberman, kind="kde");
plt.show();
sns.jointplot(x="Age", y="OperationYear", data=haberman, kind="kde");
plt.show();
sns.jointplot(x="OperationYear", y="Axil_Nodes", data=haberman, kind="kde");
plt.show();
corr = haberman.corr(method = 'spearman')
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
