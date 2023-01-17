import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
dataset=pd.read_csv("../input/rape-case-in-india/20_Victims_of_rape.csv")
dataset.head()
dataset.isnull().sum()
list_of_types=list(dataset)
# Choosing column with only Rape Crimes
import re 
r = re.compile(".*Rape.*")
Rape_Crime=list(filter(r.match, list_of_types))
# merging two dataframe
data_with_rape_cases=dataset[["Subgroup","Area_Name","Year"]].join(dataset[Rape_Crime])
# Check number of year included though its given 2014 but for sanity
data_with_rape_cases["Year"].unique()
# Dropping Year
data_with_rape_cases.drop(columns=["Year"],inplace=True)
# Grouping by state wise and adding them
data_statewise=data_with_rape_cases.groupby(["Area_Name"],as_index=False).sum()
# Plotting for each category of Rape crime individually
sns.set(style="dark")
for i in Rape_Crime:
    plt.figure(figsize=(50,15))
    plt.xlabel("Area_Name", fontsize=40)
    plt.ylabel(i, fontsize=40)
    plt.xticks(rotation=90,fontsize=30)
    plt.yticks(rotation=0,fontsize=20)
    sns.barplot(x="Area_Name",y=i,data=data_statewise)
# Adding a new column to calculate total rape cases Statewise
data_statewise["Total_Rape"]=data_statewise.iloc[:,1:].sum(axis=1)
# Plotting with total number of Rapes Statewise
plt.figure(figsize=(50,15))
plt.xlabel("Area_Name", fontsize=40)
plt.ylabel("Total_Rape", fontsize=40)
plt.xticks(rotation=90,fontsize=30)
plt.yticks(rotation=0,fontsize=20)
sns.barplot(x="Area_Name",y="Total_Rape",data=data_statewise)
# Focusing of Attempt to commit Rape
# Plotting with total number of Rapes Statewise
plt.figure(figsize=(50,15))
plt.xlabel("Area_Name", fontsize=40)
plt.ylabel("Total_Rape", fontsize=40)
plt.xticks(rotation=90,fontsize=30)
plt.yticks(rotation=0,fontsize=20)
sns.barplot(x="Area_Name",y="Rape_Cases_Reported",data=data_statewise)
