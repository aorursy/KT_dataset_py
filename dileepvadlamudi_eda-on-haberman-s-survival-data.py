import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

Habermans=pd.read_csv("../input/haberman.csv") #reading csv file
print(Habermans.shape)
labels=['Age','operation_year','axilarynodes','survival_status']
hs=pd.read_csv("../input/haberman.csv",names=labels)
print(hs.head())
print(hs.tail())
print(hs.describe())
print(hs.columns)
print(hs["survival_status"].value_counts())
hs.plot(kind='scatter',x='Age',y='operation_year');
plt.title("operation_year vs age")
plt.show()
sns.set_style("whitegrid");
sns.FacetGrid(hs,hue="survival_status",size=4)\
    .map(plt.scatter,"Age","operation_year")\
    .add_legend();
plt.title("Age vs Operaation_year")
plt.show()
sns.set_style("whitegrid");
sns.FacetGrid(hs,hue="survival_status",size=4)\
    .map(plt.scatter,"Age","axilarynodes")\
    .add_legend();
plt.title("Age vs axilarynodes")

plt.show()
plt.close()
sns.set_style("whitegrid")
sns.pairplot(hs,hue="survival_status",size=4);
plt.show()
import numpy as np
survived=hs.loc[hs["survival_status"]==1];
dead=hs.loc[hs["survival_status"]==2]
plt.plot(survived["axilarynodes"], np.zeros_like(survived['axilarynodes']), 'o')
plt.plot(dead["axilarynodes"], np.zeros_like(dead['axilarynodes']), 'o')
sns.FacetGrid(hs, hue="survival_status", size=5) \
   .map(sns.distplot, "axilarynodes") \
   .add_legend();
plt.show();
sns.FacetGrid(hs, hue="survival_status", size=5) \
   .map(sns.distplot, "operation_year") \
   .add_legend();
plt.show();
sns.FacetGrid(hs, hue="survival_status", size=5) \
   .map(sns.distplot, "Age") \
   .add_legend();
plt.show();
#Plot CDF of axilary nodes

counts, bin_edges = np.histogram(hs['axilarynodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(hs['axilarynodes'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);

plt.show();

counts, bin_edges = np.histogram(hs['axilarynodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.figure(figsize=(20,5))
for idx, feature in enumerate(list(hs.columns)[:-1]):
    plt.subplot(1, 3, idx+1)
    print("********* "+feature+" *********")
    counts, bin_edges = np.histogram(hs[feature], bins=10, density=True)
    print("Bin Edges: {}".format(bin_edges))
    pdf = counts/sum(counts)
    print("PDF: {}".format(pdf))
    cdf = np.cumsum(pdf)
    print("CDF: {}".format(cdf))
    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
    plt.xlabel(feature)
    
#survival
counts, bin_edges = np.histogram(survived['axilarynodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


# dead
counts, bin_edges = np.histogram(dead['axilarynodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
print("Means:")
print(np.mean(survived['axilarynodes']))
print(np.mean(dead["axilarynodes"]))

print("\nStd-dev:");
print(np.std(survived['axilarynodes']))
print(np.std(dead['axilarynodes']))

#Median, Quantiles, Percentiles, IQR.
print("\nMedians:")
print(np.median(survived['axilarynodes']));
print(np.median(dead['axilarynodes']))

print("\nQuantiles:")
print(np.percentile(survived['axilarynodes'],np.arange(0, 100, 25)))
print(np.percentile(dead['axilarynodes'],np.arange(0, 100, 25)))

print("\n90th Percentiles:")
print(np.percentile(survived['axilarynodes'],90))
print(np.percentile(dead['axilarynodes'],90))

from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(survived['axilarynodes']))
print(robust.mad(dead['axilarynodes']))

sns.boxplot(x='survival_status',y='axilarynodes', data=hs)
plt.show()
sns.boxplot(x='survival_status',y='Age', data=hs)
plt.show()
sns.boxplot(x='survival_status',y='operation_year', data=hs)
plt.show()
sns.violinplot(x="survival_status", y="axilarynodes", data=hs, size=8)
plt.show()
sns.violinplot(x="survival_status", y="Age", data=hs, size=8)
plt.show()
sns.violinplot(x="survival_status", y="operation_year", data=hs, size=8)
plt.show()
