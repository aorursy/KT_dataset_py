import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



# loading data from haberman.csv into a pandas datarame

haberman_df = pd.read_csv("../input/haberman.csv",names=['age', 'year_of_treatment', 'positive_lymph_nodes', 'survival_status_after_5_years'])



# Checking data is load or not  and if loaded then size of data

print(haberman_df.shape)

print(haberman_df.head())
print(haberman_df.columns)

print(haberman_df['survival_status_after_5_years'].value_counts());
# Mapping the coloumn survival_status_after_5_year into readable format

haberman_df['survival_status_after_5_years'] = haberman_df['survival_status_after_5_years'].map({1: 'survived', 2 : 'died'})
haberman_df.head()
haberman_df.info()

haberman_df['survival_status_after_5_years'].unique()
haberman_df.describe()
# Extracting rows on the basis of different class

haberman_survived = haberman_df.loc[haberman_df['survival_status_after_5_years'] == 'survived']

haberman_died = haberman_df.loc[haberman_df['survival_status_after_5_years'] == 'died']
# plotting age features 

sns.FacetGrid(haberman_df, hue= 'survival_status_after_5_years',size=5).map(sns.distplot,"age").add_legend();

plt.title("Histogram of age",fontsize=20)

plt.ylabel("Density")

plt.show()
#plotting year_of_treatment features

sns.FacetGrid(haberman_df, hue='survival_status_after_5_years',size=5).map(sns.distplot,'year_of_treatment').add_legend();

plt.title("Histogram of year_of_treatment",fontsize=20)

plt.ylabel("Density")

plt.show()
# Plotting positive_lymph_nodes Features

sns.FacetGrid(haberman_df,hue="survival_status_after_5_years", size=5).map(sns.distplot,'positive_lymph_nodes').add_legend()

plt.title("Histogram of positive_lymph_nodes",fontsize=20)

plt.ylabel("Density")

plt.show();
# Plotting of PDF and CDF of "age"



#survived data analysis 



label = ["pdf of survived data", "cdf of survived data", "pdf of died data", "cdf of died data"]

counts, bin_edges = np.histogram(haberman_survived['age'], bins = 10, density = True)

pdf = counts/sum(counts)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)



# died data analysis

counts, bin_edges = np.histogram(haberman_died['age'], bins= 10, density = True)

pdf = counts/sum(counts)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.title("PDF and CDF of 'age' feature of survived and died data analysis",fontsize=20)

plt.ylabel("Density")

plt.xlabel("Age")

plt.legend(label)

plt.show()

# Plotting of PDF and CDF of "year_of_treatment"



#survived data analysis 



counts, bin_edges = np.histogram(haberman_survived['year_of_treatment'], bins = 10, density = True)



pdf = counts/sum(counts)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)



# died data analysis

counts, bin_edges = np.histogram(haberman_died['year_of_treatment'], bins = 10, density= True)



pdf = counts/sum(counts)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.title("PDF and CDF of 'year_of_treatment' feature of survived and died data analysis",fontsize=20)

plt.ylabel("Density")

plt.xlabel("year_of_treatment")

plt.legend(label)

plt.show()
# Plotting of PDF and CDF of "positive_lymph_nodes"



#survived data analysis 



counts, bin_edges = np.histogram(haberman_survived['positive_lymph_nodes'], bins = 10, density = True)



pdf = counts/sum(counts)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)



# died data analysis

counts, bin_edges = np.histogram(haberman_died['positive_lymph_nodes'], bins = 10, density= True)



pdf = counts/sum(counts)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.title("PDF and CDF of'positive_lymph_nodes' feature of survived and died data analysis",fontsize=20)

plt.ylabel("Density")

plt.xlabel("positive_lymph_nodes")

plt.legend(label)

plt.show()
# Boxplot of "age"

sns.boxplot(x='survival_status_after_5_years', y = 'age', data= haberman_df)

plt.title("Boxplot and Whiskers of age Feature",fontsize=20)

plt.show()
# Boxplot of "year_of_treatment"

sns.boxplot(x='survival_status_after_5_years', y = 'year_of_treatment', data= haberman_df)

plt.title("Boxplot and Whiskers of year_of_treatment Feature",fontsize=20)

plt.show()
# Boxplot of "positive_lymph_nodes"

sns.boxplot(x='survival_status_after_5_years', y = 'positive_lymph_nodes', data= haberman_df)

plt.title("Boxplot and Whiskers of positive_lymph_nodes Feature")

plt.show()
# violin plot of "age"

sns.violinplot(x='survival_status_after_5_years', y = 'age', data= haberman_df)

plt.title("Violin Plots of age Feature")

plt.show()
# violin plot of "year_of_treatment"

sns.violinplot(x='survival_status_after_5_years', y = 'year_of_treatment', data= haberman_df)

plt.title("Violin Plots of year_of_treatment Feature")

plt.show()
# violin plot of "positive_lymph_nodes"

sns.violinplot(x='survival_status_after_5_years', y = 'positive_lymph_nodes', data= haberman_df)

plt.title("Violin Plots of positive_lymph_nodes Feature")

plt.show()
# Scatter Plot



sns.set_style("whitegrid");

sns.FacetGrid(haberman_df ,hue="survival_status_after_5_years",size=4).map(plt.scatter,"age","year_of_treatment").add_legend();

plt.title("BiVariate Analysis of age and year_of_treatment Feature")

plt.show()

sns.set_style("whitegrid");

sns.FacetGrid(haberman_df ,hue="survival_status_after_5_years",size=4).map(plt.scatter,"age","positive_lymph_nodes").add_legend()

plt.title("BiVariate Analysis of age and positive_lymph_nodes Feature");

plt.show()
sns.set_style("whitegrid");

sns.FacetGrid(haberman_df ,hue="survival_status_after_5_years",size=4).map(plt.scatter,"year_of_treatment","positive_lymph_nodes").add_legend();

plt.title("BiVariate Analysis of positive_lymph_nodes Feature and year_of_treatment")

plt.show()
# Pair Plot

sns.set_style("whitegrid");

sns.pairplot(haberman_df,hue="survival_status_after_5_years",vars=['age','year_of_treatment','positive_lymph_nodes'],size=3);

plt.show()