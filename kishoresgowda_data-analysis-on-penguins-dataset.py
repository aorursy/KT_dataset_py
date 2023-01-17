import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print(sns.get_dataset_names())
df=sns.load_dataset('penguins')
df.columns
df.head()
df.describe()
df.info()
df.isnull().sum()
df.dropna(inplace=True)
df['island'].value_counts().plot(kind='bar',color=['#d5e0fe','#656371','#ff7369'])
sns.set_style('white')
df['species'].value_counts().plot(kind='barh',color=['#6baddf','#01193f','#d2b486'])
sns.swarmplot(x=df.island,y=df.bill_length_mm,hue=df.species)
sns.set_style('dark')
sns.boxplot(x=df.species,y=df.body_mass_g,hue=df.sex)
sns.set_style('dark')
sns.scatterplot(x=df.bill_length_mm,y=df.bill_depth_mm,hue=df.species)
correlation_matrix=df.corr()
correlation_matrix
sns.set_style('dark')
sns.heatmap(correlation_matrix,annot=True,linecolor='white',linewidths=5,cmap="YlGnBu")
sns.set_style(style='white')
sns.pairplot(data=df,hue='species',palette=['#6baddf','#01193f','#d2b486'])