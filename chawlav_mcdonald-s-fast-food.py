#Let's first do the necessary imports

import pandas as pd

import numpy as np

from pandas import Series, DataFrame

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#Reading the data in pandas dataframe

df = pd.read_csv("../input/menu.csv")
# Setting the scale for seaborn plots

sns.set(font_scale=1.5)
#Let's have a look at the dataset

df.head()
print(df.isnull().any())
# A look at the description

df.describe()
# Unique Categories

df["Category"].unique()
# Unique Items

df["Item"].unique()
# Count plot for meal categories

g = sns.countplot(x="Category", data=df, palette="Greens_d");

g.set_xticklabels(g.get_xticklabels(), rotation=30)
x = sns.violinplot(x="Category", y="Protein", data=df)

#ax = sns.stripplot(x="Category", y="Protein", data=df, jitter = True, edgecolor="gray")



locs, labels = plt.xticks()

plt.setp(labels, rotation=45)



#I dont know how to make the [None, None, None, None...]-box go away. If you know, please feel free to write it in the comments
sns.violinplot(x="Category", data = df,y = "Calories")

locs,labels=  plt.xticks()

plt.setp(labels, rotation=45)
sns.violinplot(x="Category", data = df,y = "Sugars")

locs,labels=  plt.xticks()

plt.setp(labels, rotation=45)
#Correlation plot, heatmap to show relationship between various paramters

corr = df.corr()

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
#Co-orelation plot between calories and Protein 

sns.jointplot(x= "Calories", y = "Protein", data = df,  kind="reg", space=0, color="g")

#Co-orelation plot between calories and Carbohydrates

sns.jointplot(x= "Calories", y = "Carbohydrates", data = df,  kind="reg", space=0, color="g")

#Co-orelation plot between Total Fat and Protein 

sns.jointplot(x= "Total Fat", y = "Protein", data = df,  kind="reg", space=0, color="g") 

#Co-orelation plot between calories and Protein 

sns.jointplot(x= "Calories", y = "Protein",data = df,  kind="reg", space=0, color="g")
#Function to plot bar graphs

def barplotter(grouped):

    item = grouped["Item"].sum()

    item_list = item.sort_index()

    item_list = item_list[-20:]

    #Sizing the image canvas

    plt.figure(figsize=(8,9))

    #To plot bargraph

    g = sns.barplot(item_list.index,item_list.values)

    labels = [aj.get_text()[-40:] for aj in g.get_yticklabels()]

    g.set_yticklabels(labels)
#Plot for carbohydrates

alpha = df.groupby(df["Carbohydrates"])

barplotter(alpha)
#Plot for protein

beta = df.groupby(df["Protein"])

barplotter(beta)
#Plot for calories

gamma = df.groupby(df["Calories"])

barplotter(gamma)
#Plot for Dietary Fiber(% Daily Value)

delta = df.groupby(df["Dietary Fiber (% Daily Value)"])

barplotter(delta)
#Plot for Iron(% Daily Value)

omega = df.groupby(df["Iron (% Daily Value)"])

barplotter(delta)
#Plot for Calcium(% Daily Value)

psi = df.groupby(df["Calcium (% Daily Value)"])

barplotter(psi)
#Plot for Total Fat(% Daily Value)

omega = df.groupby(df["Total Fat"])

barplotter(omega)
