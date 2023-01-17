import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#haberman.csv into a pandas dataFrame.
haberman = pd.read_csv("../input/haberman-dataset/haberman.csv")
# to find  data-points and features
print (haberman.shape[0], " number of data points")

print (haberman.shape[1], " number of features")
haberman.columns
haberman["surv_status"].value_counts()
sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="surv_status", height=4) \
   .map(plt.scatter, "age", "axil_nodes") \
   .add_legend();
plt.show();
sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="surv_status", height=4) \
   .map(plt.scatter, "axil_nodes", "op_year") \
   .add_legend();
plt.show();
sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="surv_status", height=4) \
   .map(plt.scatter, "age", "op_year") \
   .add_legend();
plt.show();
plt.close();
sns.set_style("whitegrid");
sns.pairplot(haberman, hue = "surv_status", height=4);
plt.show()

sns.FacetGrid(haberman, hue="surv_status", height=5) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.title("Distribution plot on age for survival status")
plt.ylabel("Density")
plt.show();


sns.FacetGrid(haberman, hue="surv_status", height=5) \
   .map(sns.distplot, "op_year") \
   .add_legend();
plt.title("Distribution plot on opareation year for survival status")
plt.ylabel("Density")
plt.show();

sns.FacetGrid(haberman, hue="surv_status", height=8) \
   .map(sns.distplot, "axil_nodes") \
   .add_legend();
plt.title("Distribution plot on no. of Axil nodes identified for survival status")
plt.ylabel("Density")
plt.show();
haberman_gt_5 = haberman.loc[haberman["surv_status"] == "GT_5y"];
haberman_lt_5 = haberman.loc[haberman["surv_status"] == "LT_5y"];


labels = ["pdf of GT_5", "cdf of GT_5", "pdf of LT_5", "cdf of LT_5"]
counts, bin_edges = np.histogram(haberman_gt_5['op_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)

counts, bin_edges = np.histogram(haberman_lt_5['op_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)

plt.title("pdf and cdf on operation year")
plt.xlabel("year")
plt.ylabel("Probability")
plt.legend(labels)
plt.show();

labels = ["pdf of GT_5", "cdf of GT_5", "pdf of LT_5", "cdf of LT_5"]
counts, bin_edges = np.histogram(haberman_gt_5['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(haberman_lt_5['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)



plt.title("pdf and cdf on age")
plt.xlabel("age")
plt.ylabel("Probability")
plt.legend(labels)
plt.show();


counts, bin_edges = np.histogram(haberman_gt_5['axil_nodes'], bins=40, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(haberman_lt_5['axil_nodes'], bins=40, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)

plt.show();
sns.boxplot(x='surv_status',y='age', data=haberman)
plt.show()
sns.boxplot(x='surv_status',y='op_year', data=haberman)
plt.show()
sns.boxplot(x='surv_status',y='axil_nodes', data=haberman)
plt.show()
#Median, Quantiles, Percentiles, IQR.
print("\nMedians:")
print(np.median(haberman_gt_5["axil_nodes"]), " : GT_5y")
print(np.median(haberman_lt_5["axil_nodes"]), " : LT_5y")


print("\nDectiles:")
print(np.percentile(haberman_gt_5["axil_nodes"],np.arange(0, 100, 10)))
print(np.percentile(haberman_lt_5["axil_nodes"],np.arange(0, 100, 10)))

print("\n 5 multiple percentiles :")
print(np.percentile(haberman_gt_5["axil_nodes"],np.arange(0, 100, 5)))
print(np.percentile(haberman_lt_5["axil_nodes"],np.arange(0, 100, 5)))

from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(haberman_gt_5["axil_nodes"]))
print(robust.mad(haberman_lt_5["axil_nodes"]))

sns.violinplot(x="surv_status", y="axil_nodes", data=haberman, size=8)
plt.show()
