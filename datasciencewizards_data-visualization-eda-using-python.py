import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/sample-data/cancer_data.csv')

df.head()
#To set the figure size of plot

plt.figure(figsize=(25,10))



df['Age at diagnosis'].value_counts().plot.bar(color='red')
plt.figure(figsize=(25,10))

plt.plot(df["Age at diagnosis"])

plt.show()
plt.figure(figsize=(10,7))

plt.pie(df["Gender"].value_counts(),autopct='%1.1f%%',labels = ["male","female"])

plt.show()
sns.clustermap(df.iloc[:,:7].corr(),figsize=(15,10))
sns.boxplot(df["Age at diagnosis"])
plt.figure(figsize=(11,6))

sns.set_style("whitegrid") 

  

sns.violinplot(x = 'Gender', y = 'Age at diagnosis', data = df)
sns.swarmplot(x="Ascites degree", y="Age at diagnosis",hue="Alcohol",

              palette=["r", "c", "y"], data=df)
sns.jointplot(df["Packs of cigarets per year:"],df["Age at diagnosis"], kind="kde", height=7, space=0)