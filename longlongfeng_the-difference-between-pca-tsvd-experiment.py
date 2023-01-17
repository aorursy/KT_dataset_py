from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
%precision 3
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
data = csr_matrix((data, (row, col)), shape=(10, 10)).toarray()
print(data)
pca = PCA(n_components=5)
pca.fit_transform(data)
tsvd = TruncatedSVD(n_components=5)
tsvd.fit_transform(data)
row = np.array([0, 0, 1, 2, 2, 2, 4])
col = np.array([0, 2, 2, 0, 1, 2, 4])
data = np.array([1, 2, 3, 4, 5, 6, 10])
data = csr_matrix((data, (row, col)), shape=(10, 10)).toarray()
print(data)
pca = PCA(n_components=5)
pca.fit_transform(data)
tsvd = TruncatedSVD(n_components=5)
tsvd.fit_transform(data)
