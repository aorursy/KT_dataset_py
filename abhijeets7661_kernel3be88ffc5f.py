import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        df = pd.read_csv(os.path.join(dirname, filename))

df.head(100)

df['1.1'].value_counts()

# 1 means 224 peoples survived after operation and 2 means 81 patients died.
# Hue is our target column and 30 & 1 are my features on that we have plotted scatter plot 
sns.set_style("whitegrid");
sns.FacetGrid(df, hue="1.1", size=4).map(plt.scatter, "30", "1").add_legend();
plt.title("Scatter plot on age and axillary node  feature\n\n")
plt.show();


sns.set_style("whitegrid");
sns.FacetGrid(df, hue="1.1", size=4).map(plt.scatter, "30", "64").add_legend();
plt.title("Scatter plot on age and years feature\n\n")
plt.show();
plt.close();

sns.set_style("whitegrid");
sns.FacetGrid(df, hue="1.1", size=4).map(plt.scatter, "64", "1").add_legend();
plt.title("Scatter plot on age and years feature\n\n")
plt.show();
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="1.1", height=3);
plt.show()
sns.FacetGrid(df, hue="1.1", height=5) \
   .map(sns.distplot, "1") \
   .add_legend();
plt.show();

sns.FacetGrid(df, hue="1.1", height=5) \
   .map(sns.distplot, "30") \
   .add_legend();
plt.show();


sns.FacetGrid(df, hue="1.1", height=5) \
   .map(sns.distplot, "64") \
   .add_legend();
plt.show();


survived = df.loc[df["1.1"] == 1]
nonsur = df.loc[df["1.1"] == 2]

counts, bin_edges = np.histogram(survived['1'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

plt.show()




counts, bin_edges = np.histogram(nonsur['1'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)




plt.show();

sns.boxplot(x='1.1',y='1', data=df)
plt.show()

sns.boxplot(x='1.1',y='64', data=df)
plt.show()