# Importing all the required package 

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np

# Importing Haberman Dataset
# Renaming the columns

# 30 = Age              (Age of Patient at the time of operation)
# 64 = Operation_Year   (Year at which Operation Takes place)
# 1 = Active_Lymph      (Number of Active Lymph node)
# 1.1 = Survival_Status (Survival Status where 1 = Patient survive 5 years or more , 
#                                               2 = Patient died within 5 years)


df=pd.read_csv("../input/haberman.csv")
df.rename(columns={"30":"Age","64":"Operation_Year","1":"Active_Lymph","1.1":"Survival_Status"},inplace=True)
# Printing only some value in the dataset
print(df.head())
# Shape of data
# It tells us about number of rows and columns present in the dataset respectively.

df.shape
# To get Information About dataset

df.info()
# Converting Survival_Status into categorical data

df["Survival_Status"]=df["Survival_Status"].map({1:"Yes",2:"No"})
df.head()
# Description about Dataset

print(df.describe())
# Target Variable Distribution
print("\nTarget Variable Distribution")
print(df["Survival_Status"].value_counts())

# Normalize
print("\nTarget Variable Distribution After Normalization")
print(df["Survival_Status"].value_counts(normalize=True))
sns.set_style("whitegrid")
sns.FacetGrid(df,hue="Survival_Status",size=5).map(sns.distplot,"Age").add_legend()
plt.show()
sns.set_style("whitegrid");
sns.FacetGrid(df,hue="Survival_Status",size=5).map(sns.distplot,"Operation_Year").add_legend();
plt.show();
sns.set_style("whitegrid")
sns.FacetGrid(df,hue="Survival_Status",size=5).map(sns.distplot,"Active_Lymph").add_legend()
plt.show()
survival_less_5yrs = df[df["Survival_Status"]=="No"]
survival_more_5yrs=df[df["Survival_Status"]=="Yes"]
# Taking Age As A Parameter

counts ,bin_edges =np.histogram(survival_less_5yrs['Age'],bins=10,density=True)
pdf=counts/sum(counts)
print("Pdf And Bin_Edges For Survival Less Than 5 Years On basis of Age\n")
print(pdf)
print(bin_edges)
cdf=np.cumsum(pdf)
plt.subplot(121)
plt.grid()
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend(['Pdf Survived < 5Yrs','Cdf Survived < 5Yrs'])



counts,bin_edges = np.histogram(survival_more_5yrs['Age'],bins=10,density=True)
pdf=counts/sum(counts)
print("\n Pdf And Bin_Edges For Survival More Than 5 Years On basis of Age\n")
print(pdf)
print(bin_edges)
cdf =np.cumsum(pdf)
plt.subplot(122)
plt.grid()
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend(['Pdf Survived > 5Yrs','Cdf Survived > 5Yrs'])
plt.show()
# Taking Active Lymph As A Parameter

counts,bin_edges=np.histogram(survival_less_5yrs["Active_Lymph"],bins=10,density=True)
pdf=counts/sum(counts)
print("Pdf And Bin_Edges For Survival Less Than 5 Years On basis of Active Lymph\n")
print(pdf)
print(bin_edges)
plt.subplot(121)
plt.grid()
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend(['Pdf Survived < 5Yrs','Cdf Survived < 5yrs'])
plt.xlabel("No Of Active Lymph")
plt.ylabel("Probability")



counts,bin_edges=np.histogram(survival_more_5yrs["Active_Lymph"],bins=10,density=True)
pdf=counts/sum(counts)
print("Pdf And Bin_Edges For Survival More Than 5 Years On basis of Active Lymph\n")
print(pdf)
print(bin_edges)
plt.subplot(122)
plt.grid()
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend(['Pdf Survived > 5Yrs','Cdf Survived > 5yrs'])
plt.xlabel("No Of Active Lymph")
plt.show()
# Taking Operation Year As A parameter 

counts,bin_edges=np.histogram(survival_less_5yrs["Operation_Year"],bins=10,density=True)
pdf=counts/sum(counts)
print("Pdf And Bin_Edges For Survival Less Than 5 Years On basis of Operation Year\n")
print(pdf)
print(bin_edges)
cdf=np.cumsum(pdf)
plt.subplot(121)
plt.grid()
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend(['Pdf Survived < 5Yrs','Cdf Survived < 5yrs'])
plt.xlabel("Operation Year")
plt.ylabel("Probability")



counts,bin_edges=np.histogram(survival_more_5yrs["Operation_Year"])
pdf=counts/sum(counts)
print("Pdf And Bin_Edges For Survival More Than 5 Years On basis of Operation Year\n")
print(pdf)
print(bin_edges)
cdf=np.cumsum(pdf)
plt.subplot(122)
plt.grid()
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend(['Pdf Survived > 5Yrs','Cdf Survived > 5yrs'])
plt.xlabel("Operation Year")
plt.show()
# Taking Age As A Parameter

sns.boxplot(x="Survival_Status",y="Age",data=df)
plt.show()
# Taking Active Lymph as a Parameter
sns.boxplot(x="Survival_Status",y="Active_Lymph",data=df)
plt.show()
# Taking Operation Year as a Parameter
sns.boxplot(x="Survival_Status",y="Operation_Year",data=df)
plt.show()
# Taking Age as a parameter
sns.violinplot(x="Survival_Status",y="Age",data=df)
plt.show()
# Taking Active Lymph as a parameter

sns.violinplot(x="Survival_Status",y="Active_Lymph",data=df)
plt.show()
# Taking Operation Year as a Parameter

sns.violinplot(x="Survival_Status",y="Operation_Year",data=df)
plt.plot()
sns.set_style("whitegrid")
sns.pairplot(df,hue="Survival_Status",diag_kind="hist",size=3)
plt.show()
# Taking operation year and Active Lymph as Parameter

sns.set_style("whitegrid")
sns.FacetGrid(df,hue="Survival_Status",size=4).map(plt.scatter,"Operation_Year","Active_Lymph").add_legend()
plt.show()
# Taking Age and Active Lymph as Parameter
sns.set_style("whitegrid")
sns.FacetGrid(df,hue="Survival_Status",size=4).map(plt.scatter,"Age","Active_Lymph").add_legend()
plt.show()
