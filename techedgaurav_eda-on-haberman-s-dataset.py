# Importing all the important packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
haberman = pd.read_csv("../input/haberman.csv")
haberman.head()
# Renaming the columns for better understanding
haberman.columns = ["Age", "Op_year", "axil_nodes_det", "Survived_morethan_5years"]
haberman.head(7)
haberman["Survived_morethan_5years"] = haberman["Survived_morethan_5years"].map({1:"yes", 2:"no"})
haberman.head()
haberman.info()
haberman.describe()
haberman["Survived_morethan_5years"].value_counts()
haberman.plot(kind="scatter", x='axil_nodes_det', y='Age')
plt.grid()
plt.show()
sns.set_style("whitegrid")
sns.FacetGrid(haberman, hue="Survived_morethan_5years", size=4)\
    .map(plt.scatter, "Age", "axil_nodes_det")\
    .add_legend();
plt.show()
plt.close()
sns.set_style("whitegrid")
sns.pairplot(haberman, hue="Survived_morethan_5years", size=4)
plt.show()
sns.FacetGrid(haberman, hue="Survived_morethan_5years", size=5)\
    .map(sns.distplot, "axil_nodes_det")\
    .add_legend()
plt.show()
sns.FacetGrid(haberman, hue="Survived_morethan_5years", size=5)\
    .map(sns.distplot, "Op_year")\
    .add_legend()
plt.show()
sns.FacetGrid(haberman, hue="Survived_morethan_5years", size=5)\
    .map(sns.distplot, "Age")\
    .add_legend()
plt.show()
haberman1 = haberman.loc[haberman["Survived_morethan_5years"] == "yes"]
haberman2 = haberman.loc[haberman["Survived_morethan_5years"] == "no"]

plt.plot(haberman1["axil_nodes_det"], np.zeros_like(haberman1["axil_nodes_det"]))
plt.plot(haberman2["axil_nodes_det"], np.zeros_like(haberman2["axil_nodes_det"]),'o')
counts, bin_edges = np.histogram(haberman1['axil_nodes_det'], bins=10, density=True)

pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], cdf)
plt.plot(bin_edges[1:], pdf)

counts, bin_edges = np.histogram(haberman2['axil_nodes_det'], bins=10, density=True)

pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], cdf)
plt.plot(bin_edges[1:], pdf)

plt.show()
sns.boxplot(x='Survived_morethan_5years', y='axil_nodes_det', data=haberman)
plt.show()
sns.violinplot(x='Survived_morethan_5years', y='axil_nodes_det', data=haberman)
plt.show()
print(np.percentile(haberman1["axil_nodes_det"],90))
print(np.percentile(haberman2["axil_nodes_det"],90))
print("Mean age: ")
print(np.mean(haberman["Age"]))
