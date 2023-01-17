import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

#%matplotlib inline
dataset=pd.read_csv("../input/crime-in-india/crime/crime/42_District_wise_crimes_committed_against_women_2014.csv")
dataset.head()
dataset.isnull().sum()
list_of_types=list(dataset)
# Choosing column with only Rape Crimes

import re 

r = re.compile(".*Rape.*")

Rape_Crime=list(filter(r.match, list_of_types))
# merging two dataframe

data_with_rape_cases=dataset[["States/UTs","District","Year"]].join(dataset[Rape_Crime])
# Check number of year included though its given 2014 but for sanity

data_with_rape_cases["Year"].unique()
# Dropping Year

data_with_rape_cases.drop(columns=["Year"],inplace=True)
# Grouping by state wise and adding them

data_statewise=data_with_rape_cases.groupby(["States/UTs"],as_index=False).sum()
# Plotting for each category of Rape crime individually

sns.set(style="dark")

for i in Rape_Crime:

    plt.figure(figsize=(50,15))

    plt.xlabel("States/UTs", fontsize=40)

    plt.ylabel(i, fontsize=40)

    plt.xticks(rotation=90,fontsize=30)

    plt.yticks(rotation=0,fontsize=20)

    sns.barplot(x="States/UTs",y=i,data=data_statewise)
# Adding a new column to calculate total rape cases Statewise

data_statewise["Total_Rape"]=data_statewise.iloc[:,1:].sum(axis=1)
# Plotting with total number of Rapes Statewise

plt.figure(figsize=(50,15))

plt.xlabel("States/UTs", fontsize=40)

plt.ylabel("Total_Rape", fontsize=40)

plt.xticks(rotation=90,fontsize=30)

plt.yticks(rotation=0,fontsize=20)

sns.barplot(x="States/UTs",y="Total_Rape",data=data_statewise)
# Focusing of Attempt to commit Rape

# Plotting with total number of Rapes Statewise

plt.figure(figsize=(50,15))

plt.xlabel("States/UTs", fontsize=40)

plt.ylabel("Total_Rape", fontsize=40)

plt.xticks(rotation=90,fontsize=30)

plt.yticks(rotation=0,fontsize=20)

sns.barplot(x="States/UTs",y="Attempt to commit Rape",data=data_statewise)
# Total Rape without Attempt to commit Rape

# Adding a new column to calculate total rape cases Statewise

data_statewise["Total_Rape_excluding_attempt"]=data_statewise.iloc[:,1:-2].sum(axis=1)
# Focusing of Attempt to commit Rape

# Plotting with total number of Rapes Statewise

plt.figure(figsize=(50,15))

plt.xlabel("States/UTs", fontsize=40)

plt.ylabel("Total_Rape_excluding_attempt", fontsize=40)

plt.xticks(rotation=90,fontsize=30)

plt.yticks(rotation=0,fontsize=20)

sns.barplot(x="States/UTs",y="Total_Rape_excluding_attempt",data=data_statewise)
# These Visualisation will be updated when i will get official data of False rape accusation statewise of 2014

# If anyone have it please give a link
# Next i will find the literary rate statewise and then district wise. Does education impact these social factors

# Then filter out with female literary rate
# Next i will correlate with states with general criminal activity like theft,dacoity,arson etc

# Government policies in favour of women effects the crime rate
# A lot to be analysed in the dataset