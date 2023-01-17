import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")
df
df.describe()
sns.pairplot(df, hue = 'species')
# We're seperating the species column

species = df["species"].tolist()

X = df.drop("species", 1)
# Standardize the data

X = (X - X.mean()) / X.std(ddof=0)
# Calculating the correlation matrix of the data

X_corr = (1 / 150) * X.T.dot(X)
# Plotting the correlation matrix

plt.figure(figsize=(10,10))

sns.heatmap(X_corr, vmax=1, square=True,annot=True)

plt.title('Correlation matrix')
# method1

u,s,v = np.linalg.svd(X_corr)

eig_values, eig_vectors = s, u

eig_values, eig_vectors
# method2

np.linalg.eig(X_corr)
np.sum(eig_values)
# plotting the variance explained by each PC 

explained_variance=(eig_values / np.sum(eig_values))*100

plt.figure(figsize=(8,4))

plt.bar(range(4), explained_variance, alpha=0.6)

plt.ylabel('Percentage of explained variance')

plt.xlabel('Dimensions')
# calculating our new axis

pc1 = X.dot(eig_vectors[:,0])

pc2 = X.dot(eig_vectors[:,1])
# plotting in 2D

def plot_scatter(pc1, pc2):

    fig, ax = plt.subplots(figsize=(15, 8))

    

    species_unique = list(set(species))

    species_colors = ["r","b","g"]

    

    for i, spec in enumerate(species):

        plt.scatter(pc1[i], pc2[i], label = spec, s = 20, c=species_colors[species_unique.index(spec)])

        ax.annotate(str(i+1), (pc1[i],pc2[i]))

    

    from collections import OrderedDict

    handles, labels = plt.gca().get_legend_handles_labels()

    by_label = OrderedDict(zip(labels, handles))

    plt.legend(by_label.values(), by_label.keys(), prop={'size': 15}, loc=4)

    

    ax.set_xlabel('Principal Component 1', fontsize = 15)

    ax.set_ylabel('Principal Component 2', fontsize = 15)

    ax.axhline(y=0, color="grey", linestyle="--")

    ax.axvline(x=0, color="grey", linestyle="--")

    

    plt.grid()

    plt.axis([-4, 4, -3, 3])

    plt.show()

    

plot_scatter(pc1, pc2)
def plot_correlation_circle(pc1, pc2):    

    fig, ax = plt.subplots(figsize=(16, 10))



    for i in range(X.shape[1]):

        x = np.corrcoef(pc1,X[X.columns[i]])[0,1]

        y = np.corrcoef(pc2,X[X.columns[i]])[0,1]

        ax.annotate("", xy= (x,y), xytext=(0, 0),arrowprops=dict(arrowstyle="->"))

        ax.annotate(X.columns[i], (x+0.02,y+0.02), size=12)





    ax.set_title('Correlation circle')

    ax.axhline(y=0, color="grey", linestyle="--")

    ax.axvline(x=0, color="grey", linestyle="--")



    an = np.linspace(0, 2 * np.pi, 100)

    plt.plot(np.cos(an), np.sin(an))

    plt.axis('equal')

    plt.show()

    

plot_correlation_circle(pc1,pc2)
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
X = df.drop("species", 1)

X = StandardScaler().fit_transform(X)

pca = PCA()

result = pca.fit_transform(X)

# Remember what we said about the sign of eigen vectors that might change ?

pc1 = - result[:,0]

pc2 = - result[:,1]

plot_scatter(pc1, pc2)
import plotly.express as px
pc3 = result[:,2]
pcs = pd.DataFrame(list(zip(pc1, pc2, pc3, species)),columns =['pc1', 'pc2', 'pc3', 'species']) 

fig = px.scatter_3d(pcs, x='pc1', y='pc2', z='pc3',color='species')

fig.show()