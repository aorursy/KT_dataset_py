

#importing packages and modules

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
df=pd.read_csv("../input/haberman.csv") #importing the dataset

df.head()

df.info() #Gathering info
df["survival_status"]=df['status'].map({1:'survived',2:'dead'}) #coverting the status feature.

del df['status']
df.head() #observing the first few values of the data
df.describe() # describing the data to get statistical values
df['survival_status'].value_counts() #counting the number of people who have survived and died.
df['survival_status'].value_counts(normalize='True') # calculating the percentage of those who survived and those who died.
sns.set_style('whitegrid')

sns.FacetGrid(df,hue='survival_status',height=6).map(sns.distplot,'age')
sns.FacetGrid(df,hue='survival_status',height=6).map(sns.distplot,'nodes')
# calculating the cdf of nodes so as to determine what percentage of people have nodes below 10,20,etc...

counts,bin_edges=np.histogram(df['nodes'],bins=20,density=True)

pdf=counts/sum(counts)

print(pdf)

print(counts)
cdf=np.cumsum(pdf) #calculating the cumulative sum

plt.plot(pdf) #plotting the probability function on a normal distribution

plt.plot(cdf) #plotting the cumulative function on the same normal distribution

plt.show()

sns.boxplot(x='survival_status',y='nodes',data=df) #plotting the box plot from seaborn between nodes and survival status
sns.boxplot(x='survival_status',y='age',data=df)
sns.violinplot(x='survival_status',y='nodes',data=df)
sns.pairplot(df,hue='survival_status',height=4) #Pair plot which plots every feature in the dataset
sns.FacetGrid(df,hue='survival_status',height=6).map(plt.scatter,'age','nodes').add_legend()

plt.title('age vs nodes')