import numpy as np 

import pandas as pd

import pandas_profiling as pp

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))

%matplotlib inline
iris = pd.read_csv('../input/Iris.csv')



##Profile Report of the complete Dataset 

pp.ProfileReport(iris)
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)

sns.distplot( iris["SepalLengthCm"] , color="red", ax=axes[0, 0])

sns.distplot( iris["SepalWidthCm"] , color="yellow", ax=axes[0, 1])

sns.distplot( iris["PetalLengthCm"] , color="grey", ax=axes[1, 0])

sns.distplot( iris["PetalWidthCm"] , color="blue", ax=axes[1, 1])

unique_list = iris['Species'].unique()

unique_list
for i in unique_list:

    iris_class = iris[iris['Species'] == i]

    f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)

    plt.title('Histogram with distribution for class {}'.format(i))

    sns.distplot( iris_class["SepalLengthCm"] , color="red", ax=axes[0, 0])

    sns.distplot( iris_class["SepalWidthCm"] , color="yellow", ax=axes[0, 1])

    sns.distplot( iris_class["PetalLengthCm"] , color="grey", ax=axes[1, 0])

    sns.distplot( iris_class["PetalWidthCm"] , color="blue", ax=axes[1, 1])

    

dimen = ["PetalLengthCm", "SepalLengthCm" , "SepalWidthCm" , "PetalWidthCm"]

for i in dimen:

    sns.swarmplot(x="Species", y=i, data=iris)

    plt.show()
##Scatter of 4 choose 2 pairs of dimensions

##This signifies a pairplot with the upper triangular matrix signifying the 4c2 pairs and all the classes are color coded



new_iris= iris[iris.columns[1:6]]

sns.pairplot(new_iris, hue="Species")
from sklearn import datasets

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans



###Picking 2 principal components for fitting the Gaussian Mixture model

# Separating out the features

x = iris.iloc[:,0:5].values



# Separating out the target

y = iris.loc[:,['Species']].values



# Standardizing the features

x = StandardScaler().fit_transform(x)



pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf2 = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

principalDf2.head(5)
likelihood=[]

k=[]

from sklearn.mixture import GaussianMixture

for i in range(1,10):

    gmm = GaussianMixture(n_components=i)

    gmm.fit(principalDf2)

    labels = gmm.predict(principalDf2)

    plt.scatter(principalDf2.iloc[:, 0], principalDf2.iloc[:, 1], c=labels, s=40, cmap='viridis')

    plt.show()

    a = gmm.score(principalDf2)

    k.append(i)

    likelihood.append(a) 

    







def plot_gmm(gmm, X, label=True, ax=None):

    ax = ax or plt.gca()

    labels = gmm.fit(X).predict(X)

    if label:

        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    else:

        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], s=40, zorder=2)

    ax.axis('equal')

    

    w_factor = 0.2 / gmm.weights_.max()

    for pos, covar, w in zip(gmm.means_, gmm.covariances_ , gmm.weights_):

        draw_ellipse(pos, covar, alpha=w * w_factor)



from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):

    """Draw an ellipse with a given position and covariance"""

    ax = ax or plt.gca()

    

    # Convert covariance to principal axes

    if covariance.shape == (2, 2):

        U, s, Vt = np.linalg.svd(covariance)

        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))

        width, height = 2 * np.sqrt(s)

    else:

        angle = 0

        width, height = 2 * np.sqrt(covariance)

    

    # Draw the Ellipse

    for nsig in range(1, 4):

        ax.add_patch(Ellipse(position, nsig * width, nsig * height,

                             angle, **kwargs))



for i in range(1,10):

    gmm = GaussianMixture(n_components=i,random_state=42)

    gmm.fit(principalDf2)

    plot_gmm(gmm, principalDf2)

    plt.show()
plt.scatter(k,likelihood)

plt.title("Likelihood vs No.of Components")

plt.xlabel("Values of K")

plt.ylabel("Log-Likelihood")

plt.show