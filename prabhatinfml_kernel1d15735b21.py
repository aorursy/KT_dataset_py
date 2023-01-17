import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



import warnings 



warnings.filterwarnings("ignore") 
haberman=pd.read_csv("../input/habermans-survival-data-set/haberman.csv")
print(haberman.columns)
haberman=pd.read_csv("../input/habermans-survival-data-set/haberman.csv",header=None)
# CSV does not have meaningful headers. Adding the same



haberman=pd.read_csv("../input/habermans-survival-data-set/haberman.csv",header=None,names=["Age of patient","Year of operation","No. of Problematic nodes","Survived for 5 or more years"])
haberman.shape
haberman.info()

haberman.describe()
haberman['Survived for 5 or more years'] = haberman['Survived for 5 or more years'].map({1:"yes", 2:"no"})
# How many yes and Nos



print(haberman.iloc[:,-1].value_counts())

# Draw pair plots



plt.close()

sns.set_style("whitegrid")

pairplotGraph=sns.pairplot(haberman,hue="Survived for 5 or more years",size=5)

pairplotGraph.fig.suptitle("Haberman 2d Plot",y=1)



plt.show()


sns.set_style("whitegrid");

g=sns.FacetGrid(haberman,hue="Survived for 5 or more years",size=5).map(plt.scatter,"Year of operation","No. of Problematic nodes").add_legend()

g.fig.suptitle("Problematic Nodes Vs Year of operation")



plt.show()

sns.set_style("whitegrid");

g=sns.FacetGrid(haberman,hue="Survived for 5 or more years",size=5).map(plt.scatter,"No. of Problematic nodes","Year of operation").add_legend()

g.fig.suptitle("Rotate - Problematic Nodes Vs Year of operation")

plt.show()
import plotly.express as px

#haberman_3D = px.data.haberman()

fig = px.scatter_3d(haberman, x='Year of operation', y='No. of Problematic nodes', z='Age of patient',

              color='Survived for 5 or more years')



fig.show()
g=sns.FacetGrid(haberman,hue='Survived for 5 or more years',size=5).map(sns.distplot,"Age of patient").add_legend().set_ylabels('Count')

g.fig.suptitle("Histogram - Age of Patient")

plt.show()
g=sns.FacetGrid(haberman,hue='Survived for 5 or more years',size=5).map(sns.distplot,"Year of operation").add_legend().set_ylabels('Count')

g.fig.suptitle("Histogram - Year of operation")

plt.show()
g=sns.FacetGrid(haberman,hue='Survived for 5 or more years',size=5).map(sns.distplot,"No. of Problematic nodes").add_legend().set_ylabels('Probability')

g.fig.suptitle("Histogram - No. of Problematice Nodes")

plt.show()
haberman_Survived_for_5_or_more_years = haberman.loc[haberman["Survived for 5 or more years"]=="yes"]

haberman_Died_in_5_or_less_years = haberman.loc[haberman["Survived for 5 or more years"]=="no"]



counts,bin_edges=np.histogram(haberman_Survived_for_5_or_more_years['No. of Problematic nodes'],bins=10,density=True)



print(counts)

print(sum(counts))



pdf=counts/sum(counts)

print(pdf)

print(bin_edges)





cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)



plt.ylabel('Probability')

plt.xlabel('No. of Problematic nodes')

plt.title('Pdf/Cdf - Survived for 5 or more years')







plt.show()
# Dataset Survived_for_5_or_more_years

counts,bin_edges=np.histogram(haberman_Survived_for_5_or_more_years['No. of Problematic nodes'],bins=10,density=True)



pdf=counts/sum(counts)



cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)



# Dataset Died in 5 or less years

counts,bin_edges=np.histogram(haberman_Died_in_5_or_less_years['No. of Problematic nodes'],bins=10,density=True)



pdf=counts/sum(counts)



cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)



plt.ylabel('Probability')

plt.xlabel('No. of Problematic nodes')

plt.title('Pdf/Cdf- Haberman dataset')







plt.show()
print(np.mean(haberman_Survived_for_5_or_more_years['No. of Problematic nodes']))

print(np.mean(haberman_Died_in_5_or_less_years['No. of Problematic nodes']))



print(np.median(haberman_Survived_for_5_or_more_years['No. of Problematic nodes']))

print(np.median(haberman_Died_in_5_or_less_years['No. of Problematic nodes']))
#### Observation on Mean

#### 1. Patient who survided for 5 or more yearsis having a average of 3 problematic nodes

#### 2. Patient who died in 5 or less yearsis having an average of 7 problematic nodes



#### Observation on Median(50th percentile)

#### 1. Patient who survided for 5 or more yearsis having a 0 problematic nodes. This is with factoring any outliers

#### 2. Patient who died in 5 or less yearsis having an average of 4 problematic nodes.This is w/o facting any outliers

sns.boxplot(x='Survived for 5 or more years',y='No. of Problematic nodes',data=haberman,hue='Survived for 5 or more years',dodge=False).set_title('Overlapping View with Boxplots/Whiskers')

plt.show()
#indicate["yearsofsurvival"] = indicate["value"].isin(["Yes", "No"])



sns.violinplot(x='Survived for 5 or more years',y='No. of Problematic nodes',data=haberman,hue='Survived for 5 or more years',dodge=False).set_title('Violin View')

plt.show()
# Observations

# 1. The curve on patient who who survived for 5 or more years is having highest density for 0 problematice nodes 