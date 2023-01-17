import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



# Import the 3 dimensionality reduction methods

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA,FastICA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from scipy.linalg import svd

train = pd.read_csv('../input/train.csv')

train.head()
print(train.shape)
sample_size = train.shape[0]

dimension = train.shape[1]
# save the labels to a Pandas series target

target = train.label

# Drop the label feature

train = train.iloc[:,:-1]
train.shape
# Standardize the data

from sklearn.preprocessing import StandardScaler

X = train.values

X_std = StandardScaler().fit_transform(X)
# Calculating Eigenvectors and eigenvalues of Cov matirx

mean_vec = np.mean(X_std, axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
#check the shape of the eig_values

print(eig_vals.shape)

print(eig_vecs.shape)
# Create a list of (eigenvalue, eigenvector) tuples

eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort the eigenvalue, eigenvector pair from high to low

eig_pairs.sort(key = lambda x: x[0], reverse= True)
np.array(eig_pairs).shape
# Calculation of Explained Variance from the eigenvalues

tot = sum(eig_vals)

var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance

cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
# Invoke SKlearn's PCA method

n_components = 50

pca = PCA(n_components=n_components).fit(train.values)



eigenvalues = pca.components_.reshape(n_components, 28, 28)



eigenvalues = pca.components_
#how much each component adds up

print(pca.explained_variance_ratio_)
#total explained variance ratio

print(np.sum(pca.explained_variance_ratio_))
n_row = 5

n_col = 6



# Plot the first 8 eignenvalues

plt.figure(figsize=(15,12))

for i in list(range(n_row * n_col)):

    offset =0

    plt.subplot(n_row, n_col, i + 1)

    plt.imshow(eigenvalues[i].reshape(28,28), cmap='jet')

    title_text = 'Eigenvalue ' + str(i + 1)

    plt.title(title_text, size=10)

    plt.xticks(())

    plt.yticks(())

plt.show()
# plot some of the numbers

plt.figure(figsize=(14,12))

for digit_num in range(0,70):

    plt.subplot(7,10,digit_num+1)

    grid_data = train.iloc[digit_num].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array

    plt.imshow(grid_data, interpolation = "none", cmap = "jet")

    plt.xticks([])

    plt.yticks([])

plt.tight_layout()
# Taking only the first N rows to speed things up

X= train[:6000].values



# Standardising the values

X_std = StandardScaler().fit_transform(X)



# Call the PCA method with an explained ratio of 90%. 

pca = PCA(0.9)

pca.fit(X_std)

X_5d = pca.transform(X_std)



# For cluster coloring in our Plotly plots, remember to also restrict the target values 

Target = target[:6000]
X_5d.shape
#how much each component adds up

print(pca.explained_variance_ratio_)

#how many component do we have?

#how much each component adds up

print("we have "+str(len(pca.explained_variance_ratio_)) + " PCA components" )
#total explained variance ratio

print(np.sum(pca.explained_variance_ratio_))
principalDf = pd.DataFrame(data = X_5d[:,:2]

             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, Target], axis = 1)
finalDf.head(5)
finalDf['label'].unique()
fig = plt.figure(figsize = (18,10))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)



targets = finalDf['label'].unique()

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink','cyan','magenta']



for target, color in zip(targets,colors):

    indicesToKeep = finalDf['label'] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']

               ,finalDf.loc[indicesToKeep, 'principal component 2']

               ,c = color

               ,s = 50)

ax.legend(targets)

ax.grid()
from sklearn.cluster import KMeans # KMeans clustering 

# Set a KMeans clustering with 9 components ( 9 chosen sneakily ;) as hopefully we get back our 9 class labels)

kmeans = KMeans(n_clusters=10)

# Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(X_5d)
X_clustered[:10]
np.array(finalDf['label'])[:10]
from sklearn.metrics import classification_report



print(classification_report(finalDf['label'],X_clustered))