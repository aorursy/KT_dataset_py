import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from scipy.signal import savgol_filter

from sklearn.decomposition import PCA as sk_pca

from sklearn.preprocessing import StandardScaler

from sklearn import svm

from sklearn.cluster import KMeans
# read the data

data = pd.read_csv("../input/milk.csv")
data.head()
# the first column is the labels

lab = data.values[:, 1].astype('uint8')

# read the features (scans) and transform data from relfectance to absorbance

feat = np.log(1.0/(data.values[:, 2:]).astype('float32'))
# Calcualte first derivative applying a Savitzky-Golay filter

dfeat = savgol_filter(feat, 25, polyorder = 5, deriv = 1)
dfeat
plt.plot(feat[0], label='feature')

plt.plot(dfeat[0], label='derrivative')

plt.legend()
# Initialise

skpca1 = sk_pca(n_components = 10)

skpca2 = sk_pca(n_components = 10)



# Scale the features to have zero mean and standard deviation of 1

# This is important when correlating darta with very different variances

nfeat1 = StandardScaler().fit_transform(feat)

nfeat2 = StandardScaler().fit_transform(dfeat)
plt.plot(nfeat1[0], label='feature')

plt.plot(nfeat2[0], label='derrivative')

plt.legend()
# Fit the spectral data and extract the explained variance ratio

X1 = skpca1.fit(nfeat1)

expl_var_1 = X1.explained_variance_ratio_
print(expl_var_1)
# Fit the first derrivative data and extract the explained variance ratio

X2 = skpca2.fit(nfeat2)

expl_var_2 = X2.explained_variance_ratio_

print(expl_var_2)
# Plot data

with plt.style.context('ggplot'):

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 6))

    fig.set_tight_layout(True)

    

    ax1.plot(expl_var_1, '-o', label = "Explained Variance %")

    ax1.plot(np.cumsum(expl_var_1),'-o', label='Cumulative variance %')

    ax1.set_xlabel("PC number")

    ax1.set_title("Absobance data")

    

    ax2.plot(expl_var_2, '-o', label="Explained Variance %")

    ax2.plot(np.cumsum(expl_var_2), '-o', label='Cumulative variance %')

    ax2.set_xlabel("PC number")

    ax2.set_title("First derivative data")

    

    plt.legend()

    plt.show()
skpca2 = sk_pca(n_components=4)



# Transform on the scaled features

Xt2 = skpca2.fit_transform(nfeat2)
# Define the labels for the plot legend

labplot = [f'{i}/8 Milk' for i in range(9)]

# Scatter plot

unique = list(set(lab))

colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]

with plt.style.context('ggplot'):

    for i, u in enumerate(unique):

        col = np.expand_dims(np.array(colors[i]), axis=0)

        xi = [Xt2[j, 0] for j in range(len(Xt2[:, 0])) if lab[j] == u]

        yi = [Xt2[j, 1] for j in range(len(Xt2[:, 1])) if lab[j] == u]

        plt.scatter(xi, yi, c=col, s = 60, edgecolors='k', label=str(u))

    

    plt.xlabel('PC1')

    plt.ylabel('PC2')

    plt.legend(labplot, loc='upper right')

    plt.title('Principal Component Analysis')

    plt.show()