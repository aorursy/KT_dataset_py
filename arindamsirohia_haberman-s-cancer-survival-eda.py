# Importing Required modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Load Habermans survival dataset into pandas dataframe
haberman = pd.read_csv("../input/haberman.csv", names = ['Age','Operation Year','Auxillary Node','Survival Status'])
# How many data points and features are there in the Haberman's Dataset?
print(haberman.shape)
# What are the column names of our dataset?
print(haberman.columns)
# How many classes are present and how many data points per class are present?
haberman["Survival Status"].value_counts()

# Haberman is an imbalanced dataset as number of points are not almost equal
# Plotting distribution of Age

sns.FacetGrid(haberman, hue = "Survival Status", size = 5)\
   .map(sns.distplot, "Age")\
   .add_legend()
# Plotting distribution of Patient's Year of Operation

sns.FacetGrid(haberman, hue = "Survival Status", size = 5)\
   .map(sns.distplot, "Operation Year")\
   .add_legend()
# Plotting distribution of Number of auxillary nodes

sns.FacetGrid(haberman, hue = "Survival Status", size = 5)\
   .map(sns.distplot, "Auxillary Node")\
   .add_legend()
haberman_survived = haberman.loc[haberman["Survival Status"] == 1]
haberman_not_survived = haberman.loc[haberman["Survival Status"] == 2]
# Plots of CDF of Age for both Categories

# Survived
counts, bin_edges = np.histogram(haberman_survived['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'PDF of Age (Survived)')
plt.plot(bin_edges[1:], cdf, label = 'CDF of Age (Survived)')

# Not_Survived
counts, bin_edges = np.histogram(haberman_not_survived['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'PDF of Age (Not Survived)')
plt.plot(bin_edges[1:], cdf, label = 'CDF of Age (Not Survived)')
plt.title("PDF/CDF of Age for Survival Status")
plt.xlabel("Survival Status")
plt.legend()

plt.show()
# Plots of CDF of Operation Year for both Categories

# Survived
counts, bin_edges = np.histogram(haberman_survived['Operation Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'PDF of Operation Year (Survived)')
plt.plot(bin_edges[1:], cdf, label = 'CDF of Operation Year (Survived)')

# Not_Survived
counts, bin_edges = np.histogram(haberman_not_survived['Operation Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'PDF of Operation Year (Not Survived)')
plt.plot(bin_edges[1:], cdf, label = 'CDF of Operation Year (Not Survived)')
plt.title("PDF/CDF of Operation Year for Survival Status")
plt.xlabel("Survival Status")
plt.legend()

plt.show()
# Plots of CDF of Number of Auxillary Nodes for both Categories

# Survived
counts, bin_edges = np.histogram(haberman_survived['Auxillary Node'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'PDF of Auxillary Node (Survived)')
plt.plot(bin_edges[1:], cdf, label = 'CDF of Auxillary Node (Survived)')

# Not_Survived
counts, bin_edges = np.histogram(haberman_not_survived['Auxillary Node'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'PDF of Auxillary Node (Not Survived)')
plt.plot(bin_edges[1:], cdf, label = 'CDF of Auxillary Node (Not Survived)')
plt.title("PDF/CDF of Auxillary Node for Survival Status")
plt.xlabel("Survival Status")
plt.legend()

plt.show()
# Age

sns.boxplot(x='Survival Status',y='Age', data=haberman)
plt.show()
# Year of Operation

sns.boxplot(x='Survival Status',y='Operation Year', data=haberman)
plt.show()
# Number of Auxillary Nodes

sns.boxplot(x='Survival Status',y='Auxillary Node', data=haberman)
plt.show()
# Age

sns.violinplot(x='Survival Status',y='Age', data=haberman, size = 8)
plt.show()
# Year of Operation

sns.violinplot(x='Survival Status',y='Operation Year', data=haberman, size = 8)
plt.show()
# Number of Auxillary Nodes

sns.violinplot(x='Survival Status',y='Auxillary Node', data=haberman, size = 8)
plt.show()
# Pairwise Scatter Plot - Pair Plot

plt.close();
sns.set_style("whitegrid");
sns.pairplot(haberman, hue="Survival Status", size=3, vars = ['Age', 'Operation Year', 'Auxillary Node']);
plt.show()