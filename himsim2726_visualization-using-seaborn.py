import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../input/planet-data/Planet_data.csv')
df
sns.set_style('darkgrid')
ax = sns.barplot(data = df, ci = 5)
for label in ax.get_xticklabels()[::2]:
    label.set_visible(False)
x = df.iloc[1,1:]
sns.distplot(x)      # Plotting diameter of each planet  
sns.kdeplot(x, shade=True)
sns.boxplot(data = df)

x = df.iloc[0,1:]  # Plot for mass of planets
sns.violinplot(data=x)
sns.swarmplot(data=df)
sns.catplot(palette="YlGnBu_d", height=6, aspect=1,kind="point", data=df)
x = df.iloc[0,1:]  # Plot for mass of planets
sns.boxenplot(data = x)
x = df.iloc[0,1:] # Mass of planets
y = df.iloc[2,1:] # Density of planets
sns.scatterplot(x,y)
correlation = df.corr()
sns.heatmap(correlation, annot = True)