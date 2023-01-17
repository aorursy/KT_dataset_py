import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.decomposition import PCA as sklearnPCA



dataset = pd.read_csv('../input/combinelast2/combine_d2.csv')

dataset

x = dataset.iloc[:,0:2].values

y = dataset.iloc[:,2].values
x_std = StandardScaler().fit_transform(x)

#x_std
mean_vec= np.mean(x_std,axis=0)

cov_mat = (x_std - mean_vec).T.dot((x_std-mean_vec))/(x_std.shape[0]-1)

print('Covariance matrix \n%s' %cov_mat)
cov_mat =np.cov(x_std.T)



eig_vals, eig_vecs = np.linalg.eig(cov_mat)



print('Eigenvectors \n%a' %eig_vecs)
pca= PCA(n_components=2)

x_std = pca.fit_transform(x_std)

explained_variance= pca.explained_variance_ratio_

explained_variance
sklearn_pca = sklearnPCA(n_components = 2)

Y_sklearn = sklearn_pca.fit_transform(x_std)

#Y_sklearn
plt.figure()

plt.scatter(x, Y_sklearn)
z = dataset['Height']



plt.figure()

plt.scatter(z, y)

w = dataset['Weight']



plt.figure()

plt.scatter(w, y)
import numpy as np

import pandas as pd



data = pd.read_csv("../input/wrdata/1.csv")

data



data2 = data[["Rec", "Unnamed: 10"]]

data1 = data2[data2["Rec"] > 80]

data3 = data1[data1["Unnamed: 10"]> 800]

data
%matplotlib inline



from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale

from numpy import random, float



model = KMeans(n_clusters=6)



model = model.fit(scale(data3))



plt.figure(figsize=(8, 6))

plt.scatter(data3["Unnamed: 10"], data3["Rec"], c=model.labels_.astype(float))

plt.title('Clustering: Rec v.s Yards')

plt.xlabel('Yards')

plt.ylabel('Rec')

plt.annotate("Can't Guard Mike", (1725,149))

plt.annotate("CMC", (1005,116))

plt.annotate("Julio", (1394,99))

plt.annotate("Godwin", (1333,86))

plt.annotate("Waller", (1145,90))

plt.annotate("D. Adams", (997,83))

plt.annotate("Nuke", (1165,104))

plt.show()
