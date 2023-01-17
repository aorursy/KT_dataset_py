import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
df = pd.read_csv("../input/SAheart.csv", sep=",", decimal=".")
df.head()
df.shape
df.describe()
df.quantile(np.arange(0,1,0.1))
df.describe(include=['O'])
df['chd'].value_counts()
df['famhist'].value_counts()
pd.crosstab(index = df['famhist'], columns = df["chd"])
chd_no = df['chd'].value_counts()[0]
chd_yes = df['chd'].value_counts()[1]
np.arange(2)
plt.bar([0,1], [chd_no, chd_yes], color = ["red", "blue"])
plt.xticks([0,1], ["No","Si"])
df.boxplot()
df['age'].plot(kind='density')
df['age'].plot(kind='hist')
df.plot(kind="scatter", x = "age", y = "obesity", c = [ "red" if x=="Si" else "blue" for x in df["chd"]])
sns.pairplot(df, hue="chd", size=2.5)

grid = sns.FacetGrid(df, col='chd', row='famhist', size=2.2, aspect=1.6)
grid.map(plt.hist, 'obesity', alpha=.5)
grid.add_legend();
grid = sns.FacetGrid(df, col='chd', row='famhist', size=2.2, aspect=1.6)
grid.map(plt.hist, 'tobacco', alpha=.5)
grid.add_legend();
sns.heatmap(df.corr())
pca = PCA(n_components=2)
cmps = pca.fit_transform(df[['tobacco','age']])
plt.scatter(cmps[:,0], cmps[:,1], c=[ "red" if x=="Si" else "blue" for x in df["chd"]])
