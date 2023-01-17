# Loading the libraries

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



# Loading the data 

haber=pd.read_csv("../input/habermans-survival-data-set/haberman.csv")
#Changing the Column Names

haber.columns=['age','op_yr','aux','sur_stat']

haber.columns
# Checking the dimensions

haber.shape
#Checking the column names

haber.columns
#Checking the Catagories in Survival_Stats variable

haber['sur_stat'].unique()
#Subseting the data having class 1 in Survival_stats variable and printing the dimensions

haber_a5=haber[haber.sur_stat==1]

haber_a5.shape
#Subseting the data having class 2 in Survival_stats variable and printing the dimensions

haber_b5=haber[haber.sur_stat==2]

haber_b5.shape
# Univariate analysis on each of the variable

# Univariate analysis on the "age" variable using histogram



sns.set_style("whitegrid")

sns.FacetGrid(haber,hue="sur_stat",size=5).map(sns.distplot,"age").add_legend()
# Univariate analysis on the "op_yr" variable using histogram



sns.set_style("whitegrid")

sns.FacetGrid(haber,hue="sur_stat",size=5).map(sns.distplot,"age").add_legend()
# Univariate analysis on the "aux" variable using histogram



sns.set_style("whitegrid")

sns.FacetGrid(haber,hue="sur_stat",size=5).map(sns.distplot,"aux").add_legend()
# Using Boxplots for univariate analysis of each variable:



# Univariate analysis of "age" variable using Boxplot



sns.set_style("whitegrid")

sns.boxplot(x="sur_stat",y="age",data=haber)
# Univariate analysis of "op_yr" variable using Boxplot



sns.set_style("whitegrid")

sns.boxplot(x="sur_stat",y="op_yr",data=haber)
# Univariate analysis of "aux" variable using Boxplot



sns.set_style("whitegrid")

sns.boxplot(x="sur_stat",y="aux",data=haber)

#Univariate analysis using violin plots:



sns.set_style("whitegrid")

sns.violinplot(x="sur_stat",y="aux",data=haber,size=8)
#Bivariate analysis using 2d scatter plots (PAIR PLOTS)



sns.set_style("whitegrid")

sns.pairplot(haber,hue="sur_stat",size=2)

plt.show()
#Using pca to visualize the data

haber_data=haber.drop('sur_stat',axis=1)

haber_label=haber['sur_stat']

#print(haber_data.shape)

#print(haber_label.shape)

from sklearn.preprocessing import StandardScaler

std_data=StandardScaler().fit_transform(haber_data)

#print(std_data.shape)

from sklearn import decomposition

pca=decomposition.PCA()

pca.n_components=2

pca_data=pca.fit_transform(std_data)

#print(pca_data.shape)



final_data=np.vstack((pca_data.T,haber_label)).T

#Creating a data frame

data0=pd.DataFrame(final_data,columns=('1stDim','2ndDim','labels'))

#Visualizing the data

sns.FacetGrid(data=data0,hue='labels',height=6).map(plt.scatter,'1stDim','2ndDim').add_legend()



#Using Tsne for visualizing

from sklearn.manifold import TSNE

#Creating the model

tsne_model=TSNE(n_components=2,random_state=0,perplexity=40,n_iter=5000)

#Fiting the data to the model

data1=tsne_model.fit_transform(std_data)

#Now appending the labels to the data using vstack

data2=np.vstack((data1.T,haber_label)).T

#Now creating the dataframe for the stacked data that we made

tsne_df=pd.DataFrame(data2,columns=('1st','2nd','labels'))



#Visualizing t-sne using Seaborn

sns.FacetGrid(data=tsne_df,hue='labels',height=6).map(plt.scatter,'1st','2nd').add_legend()

plt.show()