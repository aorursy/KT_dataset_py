import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#since the csv file does not have column names so, keep header=None otherwise it will take first row as columns.
df=pd.read_csv("../input/habermans-survival-data-set/haberman.csv",header=None)
#specify the column names
df.columns=["age","operation_year","auxillary_positive_nodes","survival_status"]
df
print(df.shape)
#check the dataset is balanced or unbalanced?
df["survival_status"]=df["survival_status"].replace([1,2],["more_than_5","less_than_5"])
df["survival_status"].value_counts()
df.describe()
"""Here No.of attributes are 3 which can decide the survival_status of the patient.So,to analyse the we can plot the 3D
plot but 3D plot are somewhat difficult for analysis as it requires more mouse movement.so to analyse the data we are creating a
pair plot which are 2D""" 
sns.set_style("whitegrid")
sns.pairplot(df,hue="survival_status",height=3)
plt.show()
#lets plot histogram and pdf for each attribute
sns.FacetGrid(df,hue="survival_status",height=3).map(sns.distplot,"auxillary_positive_nodes").add_legend()
plt.show()
sns.FacetGrid(df,hue="survival_status",height=3).map(sns.distplot,"age").add_legend()
plt.show()
sns.FacetGrid(df,hue="survival_status",height=3).map(sns.distplot,"operation_year").add_legend()
plt.show()
sns.boxplot(x="survival_status",y="auxillary_positive_nodes",data=df)
plt.show()
sns.violinplot(x="survival_status",y="auxillary_positive_nodes",height=3,data=df)
plt.show()
counts,bin_edges=np.histogram(df[df["survival_status"]=="more_than_5"]["auxillary_positive_nodes"],density=True)
pdf=counts/sum(counts)
cdf = np.cumsum(pdf)

counts1,bin_edges1=np.histogram(df[df["survival_status"]=="less_than_5"]["auxillary_positive_nodes"],density=True)
pdf1=counts1/sum(counts1)
cdf1 = np.cumsum(pdf1)

plt.plot(bin_edges1[1:],pdf1,label="pdf(less_than_5)")
plt.plot(bin_edges1[1:],cdf1,label="cdf(less_than_5)")

plt.plot(bin_edges[1:],pdf,label="pdf(more_than_5)")
plt.plot(bin_edges[1:],cdf,label="cdf(more_than_5)")

plt.legend()
plt.show()
more_than_5_years=df[df["survival_status"]=="more_than_5"]
less_than_5_years=df[df["survival_status"]=="less_than_5"]
plt.figure(1)
sns.jointplot(x="age",y="auxillary_positive_nodes",data=more_than_5_years,kind="kde")
plt.title("more_than_5")
plt.figure(2)
sns.jointplot(x="age",y="auxillary_positive_nodes",data=less_than_5_years,kind="kde")
plt.title("less_than_5")
plt.show()