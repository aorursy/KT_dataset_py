from keras.datasets import mnist
import matplotlib.pyplot as plt
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
plt.imshow(X_test[90],cmap="gray")
X = X_test.reshape(-1,28*28)
Y = Y_test
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_ = sc.fit_transform(X)
plt.imshow(X_[90].reshape(28,28),cmap="gray")
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
Zpca = pca.fit_transform(X_)
pca.explained_variance_
Zpca
import numpy as np
# compute covariance matrix
covar = np.dot(X_.T,X_)
covar.shape
# Compute eigen vectors
from numpy.linalg import svd

u,s,v = svd(covar)
u.shape
ured = u[:,:2]
# projection of data on new axis (or components)

z = np.dot(X_,ured)
z
import pandas as pd

new_dataset = np.hstack((z,Y.reshape(-1,1)))
dataframe = pd.DataFrame(new_dataset, columns = ["PC1","PC2","Label"])
dataframe
import seaborn as sns

plt.figure(figsize = (15,15))
fg = sns.FacetGrid(dataframe,hue="Label",height=10)
fg.map(plt.scatter,"PC1","PC2")
fg.add_legend()
pca = PCA()
Z_pca = pca.fit_transform(X_)
Z_pca
cum_var = np.cumsum(pca.explained_variance_ratio_)
cum_var
plt.figure(figsize = (8,6))
plt.plot(cum_var)
plt.grid()
plt.xlabel("n_components")
plt.ylabel("Cimmulative Variance")
plt.show()
