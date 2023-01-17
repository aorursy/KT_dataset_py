import pandas as pd

df_wine = pd.read_csv("../input/winedata/wine.csv")

df_wine.columns = ['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', 'Magnesium','Total phenols', 'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

sc=StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)



#finding the eigen values and eigen vector using numpy

import numpy as np

cov_mat = np.cov(X_train_std.T)

eigen_vals, eigen_vacs = np.linalg.eig(cov_mat)

#Although the numpy.linalg.eig function was designed to decompose nonsymmetric square matrices, you may  nd that it returns complex eigenvalues in certain cases.

#A related function, numpy.linalg.eigh, has been implemented to decompose Hermetian matrices, which is a numerically more stable approach to work with symmetric matrices such as the covariance matrix; numpy.linalg.eigh always returns real eigenvalues.

eigen_vals

eigen_vacs[:,0]
#Using the NumPy cumsum function, we can then calculate the cumulative sum of explained variances, which we will plot via matplotlib's step function:

tot = sum(eigen_vals)

var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt

plt.bar(range(1,14), var_exp, alpha=0.5, align='center', label='individual explained variance')

plt.step(range(1,14), cum_var_exp, where = 'mid', label='cumulative align center')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')

plt.show()
eigen_pairs = [(np.abs(eigen_vals[i] ), eigen_vacs[:,i]) for i in range(len(eigen_vals))]

eigen_pairs.sort(reverse=True)

eigen_pairs
w = np.hstack((eigen_pairs[0][1][:,np.newaxis],

              eigen_pairs[1][1][:, np.newaxis]))

print('Matrix W:\n',w)
X_train_std[0].dot(w)
X_train_pca = X_train_std.dot(w)
colors = ['r', 'b','g']

markers = ['s', 'x','o']

fig = plt.figure()

ax1 = fig.add_subplot(111)

for l, c, m in zip(np.unique(y_train), colors, markers):

    ax1.scatter(X_train_pca[y_train == l, 0], X_train_pca[y_train == l,1], c=c, label=l, marker=m)

plt.xlabel('PC 1')

plt.ylabel('PC 2')

plt.legend(loc = 'lower left')

plt.show()
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map

    markers = ('s', 'x', 'o', '^', 'v')

    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())

    plt.ylim(xx2.min(), xx2.max())

    # plot class samples

    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

lr = LogisticRegression()

X_train_pca = pca.fit_transform(X_train_std)

X_test_pca = pca.transform(X_test_std)

lr.fit(X_train_pca,y_train)

plot_decision_regions(X_train_pca,y_train,classifier=lr)

plt.show()
plot_decision_regions(X_test_pca,y_test,classifier=lr)

plt.show()
pca = PCA(n_components = None)

pca.fit_transform(X_train_std)

pca.explained_variance_ratio_
np.set_printoptions(precision=4)

mean_vecs = []

for label in range(1,4):

    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))

    print(mean_vecs[label-1])
d = 13 # number of feature

S_W = np.zeros((d,d))

for label,mv in zip(range(1,4),mean_vecs):

    class_scatter = np.zeros((d,d))

    for row in X_train[y_train == label]:

        row,mv = row.reshape(d,1), mv.reshape(d,1)

        class_scatter += (row-mv).dot((row-mv).T)

    S_W += class_scatter

print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
print('Class label distribution: %s' % np.bincount(y_train)[1:])
d = 13 # number of feature

S_W = np.zeros((d,d))

for label,mv in zip(range(1,4),mean_vecs):

    class_scatter = np.cov(X_train_std[y_train == label].T)

    S_W += class_scatter

print('Scaled Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
mean_overall = np.mean(X_train_std,axis=0)

d=13# number of feature

S_B = np.zeros((d,d))

for i,mean_vec in enumerate(mean_vecs):

    n = X_train[y_train == i+1, :].shape[0]

    mean_vec = mean_vec.reshape(d,1)

    mean_overall = mean_overall.reshape(d,1)

    

S_B +=n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

#sorting eigen values in descending order

eigen_pairs = [np.abs(eigen_vals[i], eigen_vecs[:,i]) for i in range(len(eigen_vals))]

eigen_pairs = sorted(eigen_pairs,key=lambda k:k[0], reverse = True)

for eigen_vals in eigen_pairs:

    print(eigen_vals[0])

# plot the linear discriminants by decreasing eigenvalues

# tot = sum(eigen_vals.real)

# discr = [(i/tot) for i in sorted(eigen_vals.real, reverse = True)]

# cum_discr = np.cumsum(discr)

# plt.bar(range(1,14), discr, alpha = 0.5,align='center', label="individual label")

# plt.step(range(1,14), cum_discr,where='mid', label="cumulative Label")

# plt.show()

tot = sum(eigen_vals.real)

discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]

cum_discr = np.cumsum(discr)

plt.bar(range(1, 14), discr, alpha=0.5, align='center',label='individual "discriminability"')

plt.step(range(1, 14), cum_discr, where='mid',label='cumulative "discriminability"')

plt.ylabel('"discriminability" ratio')

plt.xlabel('Linear Discriminants')

plt.ylim([-0.1, 1.1])

plt.legend(loc='best')

plt.show()

#skipping the above implementation as it was getting a bit confusing.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components = 2)

X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = LogisticRegression()

lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier = lr)

plt.show()
X_test_lda = lda.transform(X_test_std)

plot_decision_regions(X_test_lda, y_test, classifier = lr)

plt.show()
from scipy.spatial.distance import pdist, squareform

from scipy import exp

from scipy.linalg import eigh

import numpy as np

def rbf_kernel_pca(X, gamma, n_components):

    sq_dists = pdist(X, 'sqeuclidean')

    mat_sq_dists = squareform(sq_dists)

    K = exp(-gamma * mat_sq_dists)

    

    # Center the kernel matrix.

    N = K.shape[0]

    one_n = np.ones((N,N)) / N

    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    

    # Obtaining eigenpairs from the centered kernel matrix

    # numpy.eigh returns them in sorted order

    eigvals, eigvecs = eigh(K)

    

    # Collect the top k eigenvectors (projected samples)

    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

    return X_pc
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100,random_state=123)

plt.scatter(X[y==0,0],X[y==0,1], color='red', marker='^', alpha=0.5)

plt.scatter(X[y==1,0],X[y==1,1],color='blue',marker='o',alpha=0.5)

plt.show()
scikit_pca = PCA(n_components = 2)

X_spca = scikit_pca.fit_transform(X)

fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))

ax[0].scatter(X_spca[y==0,0], X_spca[y==0,1], color='red', marker='^', alpha=0.5)

ax[0].scatter(X_spca[y==1,0], X_spca[y==1,1], color='blue',marker='o',alpha=0.5)

ax[1].scatter(X_spca[y==0,0], np.zeros((50,1)) + 0.02, color='red',marker='^',alpha=0.5)

ax[1].scatter(X_spca[y==1,0], np.zeros((50,1)) - 0.02, color='blue',marker='o',alpha=0.5)

plt.show()
from matplotlib.ticker import FormatStrFormatter

X_rkp = rbf_kernel_pca(X, gamma=15, n_components=2)

fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(7,3))

ax[0].scatter(X_rkp[y==0,0], X_rkp[y==0,1], color='red', marker='^', alpha=0.5)

ax[0].scatter(X_rkp[y==1,0], X_rkp[y==1,1], color='blue',marker='o',alpha=0.5)

ax[1].scatter(X_rkp[y==0,0], np.zeros((50,1)) - 0.02, color = 'red', marker='^', alpha=0.5)

ax[1].scatter(X_rkp[y==1,0], np.zeros((50,1)) + 0.02, color = 'blue', marker='o', alpha=0.5)

plt.show()
from sklearn.datasets import make_circles

X,y = make_circles(n_samples = 1000, random_state = 123, noise=0.1, factor=0.2)

plt.scatter(X[y==0,0], X[y==0,1],color='red', marker='^', alpha=0.5)

plt.scatter(X[y==1,0], X[y==1,1], color='blue', marker='o', alpha=0.5)

plt.show()
scikit_pca = PCA(n_components=2)

X_spca = scikit_pca.fit_transform(X)

fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(7,3))

ax[0].scatter(X[y==0,0], X_spca[y==0,1], color='red',marker='^', alpha=0.5)

ax[0].scatter(X[y==1,0], X_spca[y==1,1], color='blue', marker='o', alpha=0.5)



ax[1].scatter(X[y==0,0], np.zeros((500,1)) + 0.02, color='red',marker='^', alpha=0.5)

ax[1].scatter(X[y==1,0], np.zeros((500,1)) - 0.02, color='blue', marker='o', alpha=0.5)

plt.show()
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(7,3))

ax[0].scatter(X_kpca[y==0,0], X_kpca[y==0,1], color='red', marker='^', alpha=0.5)

ax[0].scatter(X_kpca[y==1,0], X_kpca[y==1,1], color='blue',marker='o',alpha=0.5)

ax[1].scatter(X_kpca[y==0,0], np.zeros((500,1)) - 0.02, color = 'red', marker='^', alpha=0.5)

ax[1].scatter(X_kpca[y==1,0], np.zeros((500,1)) + 0.02, color = 'blue', marker='o', alpha=0.5)

plt.show()
from sklearn.decomposition import KernelPCA

X, y = make_moons(n_samples = 1000, random_state=123)

scikit_kpca = KernelPCA(n_components = 2, kernel='rbf', gamma=15 )

X_sci = scikit_kpca.fit_transform(X)

plt.scatter(X_sci[y==0,0], X_sci[y==0,1], color='red', marker='^', alpha=0.5)

plt.scatter(X_sci[y==1,0], X_sci[y==1,1], color='blue', marker='o', alpha=0.5)

plt.show()
X,y = make_circles(n_samples = 1000, random_state = 123, noise=0.1, factor=0.2)

scikit_circle = KernelPCA(n_components=2, kernel='rbf', gamma=15)

X_cir = scikit_circle.fit_transform(X)

plt.scatter(X_cir[y==0,0], X_cir[y==0,1], color='red', marker='^', alpha=0.5)

plt.scatter(X_cir[y==1,0], X_cir[y==1,1], color='blue', marker='o', alpha=0.5)

plt.show()