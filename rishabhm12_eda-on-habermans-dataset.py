import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("../input/habermancsv/haberman.csv",names=["Age","Year_of_Operation","Positive_nodes","Survival_Status"])

df.head()
df.tail()

#Checking the number of datapoints in the dataset
df.shape
#Checking for balanced and imbalanced dataset
df["Survival_Status"].value_counts()
df.plot(kind="scatter",x="Age",y="Positive_nodes",ylim=(0,20))
plt.show()
df["Survival_Status"]=df["Survival_Status"].map({1:"yes",2:"no"})
df.head()
df.tail()
plt.close()
sns.set_style("whitegrid")
sns.pairplot(df,hue="Survival_Status",size=3)
plt.show()
survived_yes=df.loc[df["Survival_Status"]=="yes"]
survived_no=df.loc[df["Survival_Status"]=="no"]
plt.plot(survived_yes["Age"],np.zeros_like(survived_yes["Age"]),'g')
plt.plot(survived_no["Age"],np.zeros_like(survived_no["Age"]),'r')
plt.show()
sns.FacetGrid(df,hue="Survival_Status",size=5)\
   .map(sns.distplot,"Age")\
   .add_legend()
plt.show()
plt.close()
sns.FacetGrid(df,hue="Survival_Status",size=5)\
   .map(sns.distplot,"Positive_nodes")\
   .add_legend()
plt.show()
sns.FacetGrid(df,hue="Survival_Status",size=5)\
   .map(sns.distplot,"Year_of_Operation")\
   .add_legend()
plt.show()
counts, bin_edges = np.histogram(survived_yes['Age'], bins=10, 
                                 density = True)
pdf=counts/sum(counts)
plt.plot(bin_edges[1:],pdf)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()
counts, bin_edges = np.histogram(survived_no['Age'], bins=10, 
                                 density = True)
pdf=counts/sum(counts)
plt.close()
plt.plot(bin_edges[1:],pdf)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()
counts, bin_edges = np.histogram(survived_yes['Year_of_Operation'], bins=10, 
                                 density = True)
pdf=counts/sum(counts)
plt.close()
plt.plot(bin_edges[1:],pdf)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()
counts, bin_edges = np.histogram(survived_no['Year_of_Operation'], bins=10, 
                                 density = True)
pdf=counts/sum(counts)
plt.close()
plt.plot(bin_edges[1:],pdf)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()
print("The mean of the ages of the survived people is:")
print(np.mean(survived_yes["Age"]))
print("The mean of the number of nodes detecetd in the people who survived for more than 5 years is:")
print(np.mean(survived_yes["Positive_nodes"]))
print("The mean of the ages of the survived people is:")
print(np.mean(survived_no["Age"]))
print("The mean of the number of nodes detecetd in the people who survived for less than 5 years is:")
print(np.mean(survived_no["Positive_nodes"]))
print("Std Deviation of number of nodes")
print(np.std(survived_yes["Positive_nodes"]))
print("Std Deviation of number of nodes for people who didn't survive")
print(np.std(survived_no["Positive_nodes"]))
print("Median of ages of people who survived")
print(np.median(survived_yes["Age"]))
print("Median of ages of people who didn't survive")
print(np.median(survived_no["Age"]))
print("Median of #nodes of people who survived")
print(np.median(survived_yes["Positive_nodes"]))
print("Median of #nodes of people who didn't survive")
print(np.median(survived_no["Positive_nodes"]))
print("Median of year of operation for people who survived")
print(np.median(survived_yes["Year_of_Operation"]))
print("Median of year of operation for people who survived")
print(np.median(survived_no["Year_of_Operation"]))
print("The 0th,25th,50th and 75th percentile of the #nodes of people who survived are:")
print(np.percentile(survived_yes["Positive_nodes"],np.arange(0, 100, 25)))

print("The 0th,25th,50th and 75th percentile of the #nodes of people who didn't survive are:")
print(np.percentile(survived_no["Positive_nodes"],np.arange(0,100,25)))

print("The 0th,25th,50th and 75th percentile of the #nodes of people who survived are:")
print(np.percentile(survived_yes["Age"],np.arange(0, 100, 25)))
print("The 0th,25th,50th and 75th percentile of the #nodes of people who didn't survive are:")
print(np.percentile(survived_no["Age"],np.arange(0, 100, 25)))
#Considering the node parameter first
sns.boxplot(x='Survival_Status',y='Positive_nodes',data=df)
#Considering the year of operation
sns.boxplot(x='Survival_Status',y='Year_of_Operation',data=df)
sns.boxplot(x='Survival_Status',y='Age',data=df)
sns.violinplot(x="Survival_Status", y="Year_of_Operation",data=df, size=8)

plt.show()
sns.violinplot(x="Survival_Status", y="Age",data=df, size=8)

plt.show()




