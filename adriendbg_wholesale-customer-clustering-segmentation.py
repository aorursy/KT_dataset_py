%watermark -a "Adrien DB" -d -v -u 
%watermark --iversions
# Importing the Libraries
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import dabl
# Importing the Dataset
import os
df = pd.read_csv(r"../input/wholesale-customers-data-set/Wholesale customers data.csv")
df.head()
msno.matrix(df, figsize = (30,4))
df_data = dabl.clean(df, verbose=1)
dabl.detect_types(df_data)
df['Channel'] = df['Channel'].map({1:'Horeca', 2:'Retail'})
df['Region'].replace([1,2,3],['Lisbon','Oporto','other'],inplace=True)
def plot_distribution(df, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(df.shape[1]) / cols)
    for i, column in enumerate(df.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if df.dtypes[column] == np.object:
            g = sns.countplot(y=column, data=df)
            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            plt.xticks(rotation=25)
        else:
            g = sns.distplot(df[column])
            plt.xticks(rotation=25)
    
plot_distribution(df, cols=3, width=20, height=20, hspace=0.45, wspace=0.5)
# Let’s remove the categorical columns:
df2 = df[df.columns[+2:df.columns.size]]

#Let’s plot the distribution of each feature
def plot_distribution(df2, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(df2.shape[1]) / cols)
    for i, column in enumerate(df2.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        g = sns.boxplot(df2[column])
        plt.xticks(rotation=25)
    
plot_distribution(df2, cols=3, width=20, height=10, hspace=0.45, wspace=0.5)
sns.set(style="ticks")
g = sns.pairplot(df,corner=True,kind='reg')
g.fig.set_size_inches(15,15)
# Compute the correlation matrix
corr = df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,
            square=True, linewidths=.5, cbar_kws={"shrink": .6},annot=True)

plt.title("Pearson correlation", fontsize =20)
# First we need to convert our categorical features (region and channel) to dummy variable:
df2 = pd.get_dummies(df)
X = df2.iloc[:,:].values

sns.set()
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300,
                    n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xticks(ticks=range(1, 11))
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters = 6,
                init = 'k-means++',
                max_iter = 300,
                n_init=10,
                random_state = 0)
y_kmeans = kmeans.fit_predict(X)
df_cluster = df
df_cluster['Cluster'] = y_kmeans
df_cluster.head()
df_cluster.Cluster.value_counts()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pc = pca.fit_transform(df2)
pc_df = pd.DataFrame(pc)
pc_df.columns = ['pc1','pc2']
pca_clustering = pd.concat([pc_df,df_cluster['Cluster']],axis=1)
plt.figure(figsize=(7,7))
sns.scatterplot(x='pc1', y='pc2', hue= 'Cluster', data=pca_clustering,palette='Set1').set_title('K-Means Clustering')
plt.show()