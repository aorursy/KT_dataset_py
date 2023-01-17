# To support both python 2 and python 3

from __future__ import division, print_function, unicode_literals



# Common imports

import numpy as np

import os



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



# Where to save the figures

PROJECT_ROOT_DIR = "."

CHAPTER_ID = "unsupervised_learning"



def save_fig(fig_id, tight_layout=True):

    return

    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")

    print("Saving figure", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format='png', dpi=300)



# Ignore useless warnings (see SciPy issue #5998)

import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")
np.random.seed(4)

m = 60

w1, w2 = 0.1, 0.3

noise = 0.1



angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5

X = np.empty((m, 3))

X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2

X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2

X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)
X_centered = X - X.mean(axis=0)

U, s, Vt = np.linalg.svd(X_centered)

c1 = Vt.T[:, 0]

c2 = Vt.T[:, 1]
m, n = X.shape



S = np.zeros(X_centered.shape)

S[:n, :n] = np.diag(s)
np.allclose(X_centered, U.dot(S).dot(Vt))
W2 = Vt.T[:, :2]

X2D = X_centered.dot(W2)
X2D_using_svd = X2D
from sklearn.decomposition import PCA



pca = PCA(n_components = 2)

X2D = pca.fit_transform(X)
X2D[:5]
X2D_using_svd[:5]
np.allclose(X2D, -X2D_using_svd)
X3D_inv = pca.inverse_transform(X2D)
np.allclose(X3D_inv, X)
np.mean(np.sum(np.square(X3D_inv - X), axis=1))
X3D_inv_using_svd = X2D_using_svd.dot(Vt[:2, :])
np.allclose(X3D_inv_using_svd, X3D_inv - pca.mean_)
pca.components_
Vt[:2]
pca.explained_variance_ratio_
1 - pca.explained_variance_ratio_.sum()
np.square(s) / np.square(s).sum()
from matplotlib.patches import FancyArrowPatch

from mpl_toolkits.mplot3d import proj3d



class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):

        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)

        self._verts3d = xs, ys, zs



    def draw(self, renderer):

        xs3d, ys3d, zs3d = self._verts3d

        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)

        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        FancyArrowPatch.draw(self, renderer)
axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]



x1s = np.linspace(axes[0], axes[1], 10)

x2s = np.linspace(axes[2], axes[3], 10)

x1, x2 = np.meshgrid(x1s, x2s)



C = pca.components_

R = C.T.dot(C)

z = (R[0, 2] * x1 + R[1, 2] * x2) / (1 - R[2, 2])
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=(6, 3.8))

ax = fig.add_subplot(111, projection='3d')



X3D_above = X[X[:, 2] > X3D_inv[:, 2]]

X3D_below = X[X[:, 2] <= X3D_inv[:, 2]]



ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], "bo", alpha=0.5)



ax.plot_surface(x1, x2, z, alpha=0.2, color="k")

np.linalg.norm(C, axis=0)

ax.add_artist(Arrow3D([0, C[0, 0]],[0, C[0, 1]],[0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))

ax.add_artist(Arrow3D([0, C[1, 0]],[0, C[1, 1]],[0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))

ax.plot([0], [0], [0], "k.")



for i in range(m):

    if X[i, 2] > X3D_inv[i, 2]:

        ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-")

    else:

        ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-", color="#505050")

    

ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k+")

ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k.")

ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], "bo")

ax.set_xlabel("$x_1$", fontsize=18)

ax.set_ylabel("$x_2$", fontsize=18)

ax.set_zlabel("$x_3$", fontsize=18)

ax.set_xlim(axes[0:2])

ax.set_ylim(axes[2:4])

ax.set_zlim(axes[4:6])



# Note: If you are using Matplotlib 3.0.0, it has a bug and does not

# display 3D graphs properly.

# See https://github.com/matplotlib/matplotlib/issues/12239

# You should upgrade to a later version. If you cannot, then you can

# use the following workaround before displaying each 3D graph:

# for spine in ax.spines.values():

#     spine.set_visible(False)



save_fig("dataset_3d_plot")

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, aspect='equal')



ax.plot(X2D[:, 0], X2D[:, 1], "k+")

ax.plot(X2D[:, 0], X2D[:, 1], "k.")

ax.plot([0], [0], "ko")

ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')

ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')

ax.set_xlabel("$z_1$", fontsize=18)

ax.set_ylabel("$z_2$", fontsize=18, rotation=0)

ax.axis([-1.5, 1.3, -1.2, 1.2])

ax.grid(True)

save_fig("dataset_2d_plot")
from sklearn.datasets import make_swiss_roll

X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
axes = [-11.5, 14, -2, 23, -12, 15]



fig = plt.figure(figsize=(6, 5))

ax = fig.add_subplot(111, projection='3d')



ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)

ax.view_init(10, -70)

ax.set_xlabel("$x_1$", fontsize=18)

ax.set_ylabel("$x_2$", fontsize=18)

ax.set_zlabel("$x_3$", fontsize=18)

ax.set_xlim(axes[0:2])

ax.set_ylim(axes[2:4])

ax.set_zlim(axes[4:6])



save_fig("swiss_roll_plot")

plt.show()
plt.figure(figsize=(11, 4))



plt.subplot(121)

plt.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.hot)

plt.axis(axes[:4])

plt.xlabel("$x_1$", fontsize=18)

plt.ylabel("$x_2$", fontsize=18, rotation=0)

plt.grid(True)



plt.subplot(122)

plt.scatter(t, X[:, 1], c=t, cmap=plt.cm.hot)

plt.axis([4, 15, axes[2], axes[3]])

plt.xlabel("$z_1$", fontsize=18)

plt.grid(True)



save_fig("squished_swiss_roll_plot")

plt.show()
from matplotlib import gridspec



axes = [-11.5, 14, -2, 23, -12, 15]



x2s = np.linspace(axes[2], axes[3], 10)

x3s = np.linspace(axes[4], axes[5], 10)

x2, x3 = np.meshgrid(x2s, x3s)



fig = plt.figure(figsize=(6, 5))

ax = plt.subplot(111, projection='3d')



positive_class = X[:, 0] > 5

X_pos = X[positive_class]

X_neg = X[~positive_class]

ax.view_init(10, -70)

ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")

ax.plot_wireframe(5, x2, x3, alpha=0.5)

ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")

ax.set_xlabel("$x_1$", fontsize=18)

ax.set_ylabel("$x_2$", fontsize=18)

ax.set_zlabel("$x_3$", fontsize=18)

ax.set_xlim(axes[0:2])

ax.set_ylim(axes[2:4])

ax.set_zlim(axes[4:6])



save_fig("manifold_decision_boundary_plot1")

plt.show()



fig = plt.figure(figsize=(5, 4))

ax = plt.subplot(111)



plt.plot(t[positive_class], X[positive_class, 1], "gs")

plt.plot(t[~positive_class], X[~positive_class, 1], "y^")

plt.axis([4, 15, axes[2], axes[3]])

plt.xlabel("$z_1$", fontsize=18)

plt.ylabel("$z_2$", fontsize=18, rotation=0)

plt.grid(True)



save_fig("manifold_decision_boundary_plot2")

plt.show()



fig = plt.figure(figsize=(6, 5))

ax = plt.subplot(111, projection='3d')



positive_class = 2 * (t[:] - 4) > X[:, 1]

X_pos = X[positive_class]

X_neg = X[~positive_class]

ax.view_init(10, -70)

ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")

ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")

ax.set_xlabel("$x_1$", fontsize=18)

ax.set_ylabel("$x_2$", fontsize=18)

ax.set_zlabel("$x_3$", fontsize=18)

ax.set_xlim(axes[0:2])

ax.set_ylim(axes[2:4])

ax.set_zlim(axes[4:6])



save_fig("manifold_decision_boundary_plot3")

plt.show()



fig = plt.figure(figsize=(5, 4))

ax = plt.subplot(111)



plt.plot(t[positive_class], X[positive_class, 1], "gs")

plt.plot(t[~positive_class], X[~positive_class, 1], "y^")

plt.plot([4, 15], [0, 22], "b-", linewidth=2)

plt.axis([4, 15, axes[2], axes[3]])

plt.xlabel("$z_1$", fontsize=18)

plt.ylabel("$z_2$", fontsize=18, rotation=0)

plt.grid(True)



save_fig("manifold_decision_boundary_plot4")

plt.show()
angle = np.pi / 5

stretch = 5

m = 200



np.random.seed(3)

X = np.random.randn(m, 2) / 10

X = X.dot(np.array([[stretch, 0],[0, 1]])) # stretch

X = X.dot([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]) # rotate



u1 = np.array([np.cos(angle), np.sin(angle)])

u2 = np.array([np.cos(angle - 2 * np.pi/6), np.sin(angle - 2 * np.pi/6)])

u3 = np.array([np.cos(angle - np.pi/2), np.sin(angle - np.pi/2)])



X_proj1 = X.dot(u1.reshape(-1, 1))

X_proj2 = X.dot(u2.reshape(-1, 1))

X_proj3 = X.dot(u3.reshape(-1, 1))



plt.figure(figsize=(8,4))

plt.subplot2grid((3,2), (0, 0), rowspan=3)

plt.plot([-1.4, 1.4], [-1.4*u1[1]/u1[0], 1.4*u1[1]/u1[0]], "k-", linewidth=1)

plt.plot([-1.4, 1.4], [-1.4*u2[1]/u2[0], 1.4*u2[1]/u2[0]], "k--", linewidth=1)

plt.plot([-1.4, 1.4], [-1.4*u3[1]/u3[0], 1.4*u3[1]/u3[0]], "k:", linewidth=2)

plt.plot(X[:, 0], X[:, 1], "bo", alpha=0.5)

plt.axis([-1.4, 1.4, -1.4, 1.4])

plt.arrow(0, 0, u1[0], u1[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')

plt.arrow(0, 0, u3[0], u3[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')

plt.text(u1[0] + 0.1, u1[1] - 0.05, r"$\mathbf{c_1}$", fontsize=22)

plt.text(u3[0] + 0.1, u3[1], r"$\mathbf{c_2}$", fontsize=22)

plt.xlabel("$x_1$", fontsize=18)

plt.ylabel("$x_2$", fontsize=18, rotation=0)

plt.grid(True)



plt.subplot2grid((3,2), (0, 1))

plt.plot([-2, 2], [0, 0], "k-", linewidth=1)

plt.plot(X_proj1[:, 0], np.zeros(m), "bo", alpha=0.3)

plt.gca().get_yaxis().set_ticks([])

plt.gca().get_xaxis().set_ticklabels([])

plt.axis([-2, 2, -1, 1])

plt.grid(True)



plt.subplot2grid((3,2), (1, 1))

plt.plot([-2, 2], [0, 0], "k--", linewidth=1)

plt.plot(X_proj2[:, 0], np.zeros(m), "bo", alpha=0.3)

plt.gca().get_yaxis().set_ticks([])

plt.gca().get_xaxis().set_ticklabels([])

plt.axis([-2, 2, -1, 1])

plt.grid(True)



plt.subplot2grid((3,2), (2, 1))

plt.plot([-2, 2], [0, 0], "k:", linewidth=2)

plt.plot(X_proj3[:, 0], np.zeros(m), "bo", alpha=0.3)

plt.gca().get_yaxis().set_ticks([])

plt.axis([-2, 2, -1, 1])

plt.xlabel("$z_1$", fontsize=18)

plt.grid(True)



save_fig("pca_best_projection")

plt.show()
from six.moves import urllib

try:

    from sklearn.datasets import fetch_openml

    mnist = fetch_openml('mnist_784', version=1)

    mnist.target = mnist.target.astype(np.int64)

except ImportError:

    from sklearn.datasets import fetch_mldata

    mnist = fetch_mldata('MNIST original')
from sklearn.model_selection import train_test_split



X = mnist["data"]

y = mnist["target"]



X_train, X_test, y_train, y_test = train_test_split(X, y)
pca = PCA()

pca.fit(X_train)

cumsum = np.cumsum(pca.explained_variance_ratio_)

d = np.argmax(cumsum >= 0.95) + 1
d
pca = PCA(n_components=0.95)

X_reduced = pca.fit_transform(X_train)
pca.n_components_
np.sum(pca.explained_variance_ratio_)
pca = PCA(n_components = 154)

X_reduced = pca.fit_transform(X_train)

X_recovered = pca.inverse_transform(X_reduced)
def plot_digits(instances, images_per_row=5, **options):

    size = 28

    images_per_row = min(len(instances), images_per_row)

    images = [instance.reshape(size,size) for instance in instances]

    n_rows = (len(instances) - 1) // images_per_row + 1

    row_images = []

    n_empty = n_rows * images_per_row - len(instances)

    images.append(np.zeros((size, size * n_empty)))

    for row in range(n_rows):

        rimages = images[row * images_per_row : (row + 1) * images_per_row]

        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)

    plt.imshow(image, cmap = mpl.cm.binary, **options)

    plt.axis("off")
plt.figure(figsize=(7, 4))

plt.subplot(121)

plot_digits(X_train[::2100])

plt.title("Original", fontsize=16)

plt.subplot(122)

plot_digits(X_recovered[::2100])

plt.title("Compressed", fontsize=16)



save_fig("mnist_compression_plot")
X_reduced_pca = X_reduced
from sklearn.decomposition import IncrementalPCA



n_batches = 100

inc_pca = IncrementalPCA(n_components=154)

for X_batch in np.array_split(X_train, n_batches):

    print(".", end="") # not shown in the book

    inc_pca.partial_fit(X_batch)



X_reduced = inc_pca.transform(X_train)
X_recovered_inc_pca = inc_pca.inverse_transform(X_reduced)
plt.figure(figsize=(7, 4))

plt.subplot(121)

plot_digits(X_train[::2100])

plt.subplot(122)

plot_digits(X_recovered_inc_pca[::2100])

plt.tight_layout()
X_reduced_inc_pca = X_reduced
np.allclose(pca.mean_, inc_pca.mean_)
np.allclose(X_reduced_pca, X_reduced_inc_pca)
filename = "my_mnist.data"

m, n = X_train.shape



X_mm = np.memmap(filename, dtype='float32', mode='write', shape=(m, n))

X_mm[:] = X_train
del X_mm
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))



batch_size = m // n_batches

inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)

inc_pca.fit(X_mm)
rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)

X_reduced = rnd_pca.fit_transform(X_train)
import time



for n_components in (2, 10, 154):

    print("n_components =", n_components)

    regular_pca = PCA(n_components=n_components)

    inc_pca = IncrementalPCA(n_components=n_components, batch_size=500)

    rnd_pca = PCA(n_components=n_components, random_state=42, svd_solver="randomized")



    for pca in (regular_pca, inc_pca, rnd_pca):

        t1 = time.time()

        pca.fit(X_train)

        t2 = time.time()

        print("    {}: {:.1f} seconds".format(pca.__class__.__name__, t2 - t1))
times_rpca = []

times_pca = []

sizes = [1000, 10000, 20000, 30000, 40000, 50000, 70000, 100000, 200000, 500000]

for n_samples in sizes:

    X = np.random.randn(n_samples, 5)

    pca = PCA(n_components = 2, svd_solver="randomized", random_state=42)

    t1 = time.time()

    pca.fit(X)

    t2 = time.time()

    times_rpca.append(t2 - t1)

    pca = PCA(n_components = 2)

    t1 = time.time()

    pca.fit(X)

    t2 = time.time()

    times_pca.append(t2 - t1)



plt.plot(sizes, times_rpca, "b-o", label="RPCA")

plt.plot(sizes, times_pca, "r-s", label="PCA")

plt.xlabel("n_samples")

plt.ylabel("Training time")

plt.legend(loc="upper left")

plt.title("PCA and Randomized PCA time complexity ")
times_rpca = []

times_pca = []

sizes = [1000, 2000, 3000, 4000, 5000, 6000]

for n_features in sizes:

    X = np.random.randn(2000, n_features)

    pca = PCA(n_components = 2, random_state=42, svd_solver="randomized")

    t1 = time.time()

    pca.fit(X)

    t2 = time.time()

    times_rpca.append(t2 - t1)

    pca = PCA(n_components = 2)

    t1 = time.time()

    pca.fit(X)

    t2 = time.time()

    times_pca.append(t2 - t1)



plt.plot(sizes, times_rpca, "b-o", label="RPCA")

plt.plot(sizes, times_pca, "r-s", label="PCA")

plt.xlabel("n_features")

plt.ylabel("Training time")

plt.legend(loc="upper left")

plt.title("PCA and Randomized PCA time complexity ")
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
from sklearn.decomposition import KernelPCA



rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)

X_reduced = rbf_pca.fit_transform(X)
from sklearn.decomposition import KernelPCA



lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)

sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)



y = t > 6.9



plt.figure(figsize=(11, 4))

for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):

    X_reduced = pca.fit_transform(X)

    if subplot == 132:

        X_reduced_rbf = X_reduced

    

    plt.subplot(subplot)

    #plt.plot(X_reduced[y, 0], X_reduced[y, 1], "gs")

    #plt.plot(X_reduced[~y, 0], X_reduced[~y, 1], "y^")

    plt.title(title, fontsize=14)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)

    plt.xlabel("$z_1$", fontsize=18)

    if subplot == 131:

        plt.ylabel("$z_2$", fontsize=18, rotation=0)

    plt.grid(True)



save_fig("kernel_pca_plot")

plt.show()
plt.figure(figsize=(6, 5))



X_inverse = rbf_pca.inverse_transform(X_reduced_rbf)



ax = plt.subplot(111, projection='3d')

ax.view_init(10, -70)

ax.scatter(X_inverse[:, 0], X_inverse[:, 1], X_inverse[:, 2], c=t, cmap=plt.cm.hot, marker="x")

ax.set_xlabel("")

ax.set_ylabel("")

ax.set_zlabel("")

ax.set_xticklabels([])

ax.set_yticklabels([])

ax.set_zticklabels([])



save_fig("preimage_plot", tight_layout=False)

plt.show()
X_reduced = rbf_pca.fit_transform(X)



plt.figure(figsize=(11, 4))

plt.subplot(132)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot, marker="x")

plt.xlabel("$z_1$", fontsize=18)

plt.ylabel("$z_2$", fontsize=18, rotation=0)

plt.grid(True)
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline



clf = Pipeline([

        ("kpca", KernelPCA(n_components=2)),

        ("log_reg", LogisticRegression(solver="liblinear"))

    ])



param_grid = [{

        "kpca__gamma": np.linspace(0.03, 0.05, 10),

        "kpca__kernel": ["rbf", "sigmoid"]

    }]



grid_search = GridSearchCV(clf, param_grid, cv=3)

grid_search.fit(X, y)
print(grid_search.best_params_)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433,

                    fit_inverse_transform=True)

X_reduced = rbf_pca.fit_transform(X)

X_preimage = rbf_pca.inverse_transform(X_reduced)
from sklearn.metrics import mean_squared_error



mean_squared_error(X, X_preimage)
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)
from sklearn.manifold import LocallyLinearEmbedding



lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)

X_reduced = lle.fit_transform(X)
plt.title("Unrolled swiss roll using LLE", fontsize=14)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)

plt.xlabel("$z_1$", fontsize=18)

plt.ylabel("$z_2$", fontsize=18)

plt.axis([-0.065, 0.055, -0.1, 0.12])

plt.grid(True)



save_fig("lle_unrolling_plot")

plt.show()
from sklearn.manifold import MDS



mds = MDS(n_components=2, random_state=42)

X_reduced_mds = mds.fit_transform(X)
from sklearn.manifold import Isomap



isomap = Isomap(n_components=2)

X_reduced_isomap = isomap.fit_transform(X)
from sklearn.manifold import TSNE



tsne = TSNE(n_components=2, random_state=42)

X_reduced_tsne = tsne.fit_transform(X)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



lda = LinearDiscriminantAnalysis(n_components=2)

X_mnist = mnist["data"]

y_mnist = mnist["target"]

lda.fit(X_mnist, y_mnist)

X_reduced_lda = lda.transform(X_mnist)
titles = ["MDS", "Isomap", "t-SNE"]



plt.figure(figsize=(11,4))



for subplot, title, X_reduced in zip((131, 132, 133), titles,

                                     (X_reduced_mds, X_reduced_isomap, X_reduced_tsne)):

    plt.subplot(subplot)

    plt.title(title, fontsize=14)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)

    plt.xlabel("$z_1$", fontsize=18)

    if subplot == 131:

        plt.ylabel("$z_2$", fontsize=18, rotation=0)

    plt.grid(True)



save_fig("other_dim_reduction_plot")

plt.show()
def learned_parameters(model):

    return [m for m in dir(model)

            if m.endswith("_") and not m.startswith("_")]
from sklearn.datasets import load_iris
data = load_iris()

X = data.data

y = data.target

data.target_names
plt.figure(figsize=(9, 3.5))



plt.subplot(121)

plt.plot(X[y==0, 2], X[y==0, 3], "yo", label="Iris-Setosa")

plt.plot(X[y==1, 2], X[y==1, 3], "bs", label="Iris-Versicolor")

plt.plot(X[y==2, 2], X[y==2, 3], "g^", label="Iris-Virginica")

plt.xlabel("Petal length", fontsize=14)

plt.ylabel("Petal width", fontsize=14)

plt.legend(fontsize=12)



plt.subplot(122)

plt.scatter(X[:, 2], X[:, 3], c="k", marker=".")

plt.xlabel("Petal length", fontsize=14)

plt.tick_params(labelleft=False)



save_fig("classification_vs_clustering_diagram")

plt.show()
from sklearn.mixture import GaussianMixture
y_pred = GaussianMixture(n_components=3, random_state=42).fit(X).predict(X)

mapping = np.array([2, 0, 1])

y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])
plt.plot(X[y_pred==0, 2], X[y_pred==0, 3], "yo", label="Cluster 1")

plt.plot(X[y_pred==1, 2], X[y_pred==1, 3], "bs", label="Cluster 2")

plt.plot(X[y_pred==2, 2], X[y_pred==2, 3], "g^", label="Cluster 3")

plt.xlabel("Petal length", fontsize=14)

plt.ylabel("Petal width", fontsize=14)

plt.legend(loc="upper left", fontsize=12)

plt.show()
np.sum(y_pred==y)
np.sum(y_pred==y) / len(y_pred)
from sklearn.datasets import make_blobs
blob_centers = np.array(

    [[ 0.2,  2.3],

     [-1.5 ,  2.3],

     [-2.8,  1.8],

     [-2.8,  2.8],

     [-2.8,  1.3]])

blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centers,

                  cluster_std=blob_std, random_state=7)
def plot_clusters(X, y=None):

    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)

    plt.xlabel("$x_1$", fontsize=14)

    plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.figure(figsize=(8, 4))

plot_clusters(X)

save_fig("blobs_diagram")

plt.show()
from sklearn.cluster import KMeans
k = 5

kmeans = KMeans(n_clusters=k, random_state=42)

y_pred = kmeans.fit_predict(X)
y_pred
y_pred is kmeans.labels_
kmeans.cluster_centers_
kmeans.labels_
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])

kmeans.predict(X_new)
def plot_data(X):

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)



def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):

    if weights is not None:

        centroids = centroids[weights > weights.max() / 10]

    plt.scatter(centroids[:, 0], centroids[:, 1],

                marker='o', s=30, linewidths=8,

                color=circle_color, zorder=10, alpha=0.9)

    plt.scatter(centroids[:, 0], centroids[:, 1],

                marker='x', s=50, linewidths=50,

                color=cross_color, zorder=11, alpha=1)



def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,

                             show_xlabels=True, show_ylabels=True):

    mins = X.min(axis=0) - 0.1

    maxs = X.max(axis=0) + 0.1

    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),

                         np.linspace(mins[1], maxs[1], resolution))

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)



    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),

                cmap="Pastel2")

    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),

                linewidths=1, colors='k')

    plot_data(X)

    if show_centroids:

        plot_centroids(clusterer.cluster_centers_)



    if show_xlabels:

        plt.xlabel("$x_1$", fontsize=14)

    else:

        plt.tick_params(labelbottom=False)

    if show_ylabels:

        plt.ylabel("$x_2$", fontsize=14, rotation=0)

    else:

        plt.tick_params(labelleft=False)
plt.figure(figsize=(8, 4))

plot_decision_boundaries(kmeans, X)

save_fig("voronoi_diagram")

plt.show()
kmeans.transform(X_new)
np.linalg.norm(np.tile(X_new, (1, k)).reshape(-1, k, 2) - kmeans.cluster_centers_, axis=2)
kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1,

                     algorithm="full", max_iter=1, random_state=1)

kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1,

                     algorithm="full", max_iter=2, random_state=1)

kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1,

                     algorithm="full", max_iter=3, random_state=1)

kmeans_iter1.fit(X)

kmeans_iter2.fit(X)

kmeans_iter3.fit(X)
plt.figure(figsize=(10, 8))



plt.subplot(321)

plot_data(X)

plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')

plt.ylabel("$x_2$", fontsize=14, rotation=0)

plt.tick_params(labelbottom=False)

plt.title("Update the centroids (initially randomly)", fontsize=14)



plt.subplot(322)

plot_decision_boundaries(kmeans_iter1, X, show_xlabels=False, show_ylabels=False)

plt.title("Label the instances", fontsize=14)



plt.subplot(323)

plot_decision_boundaries(kmeans_iter1, X, show_centroids=False, show_xlabels=False)

plot_centroids(kmeans_iter2.cluster_centers_)



plt.subplot(324)

plot_decision_boundaries(kmeans_iter2, X, show_xlabels=False, show_ylabels=False)



plt.subplot(325)

plot_decision_boundaries(kmeans_iter2, X, show_centroids=False)

plot_centroids(kmeans_iter3.cluster_centers_)



plt.subplot(326)

plot_decision_boundaries(kmeans_iter3, X, show_ylabels=False)



save_fig("kmeans_algorithm_diagram")

plt.show()
def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):

    clusterer1.fit(X)

    clusterer2.fit(X)



    plt.figure(figsize=(10, 3.2))



    plt.subplot(121)

    plot_decision_boundaries(clusterer1, X)

    if title1:

        plt.title(title1, fontsize=14)



    plt.subplot(122)

    plot_decision_boundaries(clusterer2, X, show_ylabels=False)

    if title2:

        plt.title(title2, fontsize=14)
kmeans_rnd_init1 = KMeans(n_clusters=5, init="random", n_init=1,

                         algorithm="full", random_state=11)

kmeans_rnd_init2 = KMeans(n_clusters=5, init="random", n_init=1,

                         algorithm="full", random_state=19)



plot_clusterer_comparison(kmeans_rnd_init1, kmeans_rnd_init2, X,

                          "Solution 1", "Solution 2 (with a different random init)")



save_fig("kmeans_variability_diagram")

plt.show()
kmeans.inertia_
X_dist = kmeans.transform(X)

np.sum(X_dist[np.arange(len(X_dist)), kmeans.labels_]**2)
kmeans.score(X)
kmeans_rnd_init1.inertia_
kmeans_rnd_init2.inertia_
kmeans_rnd_10_inits = KMeans(n_clusters=5, init="random", n_init=10,

                              algorithm="full", random_state=11)

kmeans_rnd_10_inits.fit(X)
plt.figure(figsize=(8, 4))

plot_decision_boundaries(kmeans_rnd_10_inits, X)

plt.show()
KMeans()
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])

kmeans = KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)

kmeans.fit(X)

kmeans.inertia_
%timeit -n 50 KMeans(algorithm="elkan").fit(X)
%timeit -n 50 KMeans(algorithm="full").fit(X)
from sklearn.cluster import MiniBatchKMeans
minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)

minibatch_kmeans.fit(X)
minibatch_kmeans.inertia_
filename = "my_mnist.data"

m, n = 50000, 28*28

X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
minibatch_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10, random_state=42)

minibatch_kmeans.fit(X_mm)
def load_next_batch(batch_size):

    return X[np.random.choice(len(X), batch_size, replace=False)]
np.random.seed(42)
k = 5

n_init = 10

n_iterations = 100

batch_size = 100

init_size = 500  # more data for K-Means++ initialization

evaluate_on_last_n_iters = 10



best_kmeans = None



for init in range(n_init):

    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, init_size=init_size)

    X_init = load_next_batch(init_size)

    minibatch_kmeans.partial_fit(X_init)



    minibatch_kmeans.sum_inertia_ = 0

    for iteration in range(n_iterations):

        X_batch = load_next_batch(batch_size)

        minibatch_kmeans.partial_fit(X_batch)

        if iteration >= n_iterations - evaluate_on_last_n_iters:

            minibatch_kmeans.sum_inertia_ += minibatch_kmeans.inertia_



    if (best_kmeans is None or

        minibatch_kmeans.sum_inertia_ < best_kmeans.sum_inertia_):

        best_kmeans = minibatch_kmeans
best_kmeans.score(X)
%timeit KMeans(n_clusters=5).fit(X)
%timeit MiniBatchKMeans(n_clusters=5).fit(X)
from timeit import timeit
times = np.empty((100, 2))

inertias = np.empty((100, 2))

for k in range(1, 101):

    kmeans = KMeans(n_clusters=k, random_state=42)

    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)

    print("\r{}/{}".format(k, 100), end="")

    times[k-1, 0] = timeit("kmeans.fit(X)", number=10, globals=globals())

    times[k-1, 1]  = timeit("minibatch_kmeans.fit(X)", number=10, globals=globals())

    inertias[k-1, 0] = kmeans.inertia_

    inertias[k-1, 1] = minibatch_kmeans.inertia_
plt.figure(figsize=(10,4))



plt.subplot(121)

plt.plot(range(1, 101), inertias[:, 0], "r--", label="K-Means")

plt.plot(range(1, 101), inertias[:, 1], "b.-", label="Mini-batch K-Means")

plt.xlabel("$k$", fontsize=16)

#plt.ylabel("Inertia", fontsize=14)

plt.title("Inertia", fontsize=14)

plt.legend(fontsize=14)

plt.axis([1, 100, 0, 100])



plt.subplot(122)

plt.plot(range(1, 101), times[:, 0], "r--", label="K-Means")

plt.plot(range(1, 101), times[:, 1], "b.-", label="Mini-batch K-Means")

plt.xlabel("$k$", fontsize=16)

#plt.ylabel("Training time (seconds)", fontsize=14)

plt.title("Training time (seconds)", fontsize=14)

plt.axis([1, 100, 0, 6])

#plt.legend(fontsize=14)



save_fig("minibatch_kmeans_vs_kmeans")

plt.show()
kmeans_k3 = KMeans(n_clusters=3, random_state=42)

kmeans_k8 = KMeans(n_clusters=8, random_state=42)



plot_clusterer_comparison(kmeans_k3, kmeans_k8, X, "$k=3$", "$k=8$")

save_fig("bad_n_clusters_diagram")

plt.show()
kmeans_k3.inertia_
kmeans_k8.inertia_
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)

                for k in range(1, 10)]

inertias = [model.inertia_ for model in kmeans_per_k]
plt.figure(figsize=(8, 3.5))

plt.plot(range(1, 10), inertias, "bo-")

plt.xlabel("$k$", fontsize=14)

plt.ylabel("Inertia", fontsize=14)

plt.annotate('Elbow',

             xy=(4, inertias[3]),

             xytext=(0.55, 0.55),

             textcoords='figure fraction',

             fontsize=16,

             arrowprops=dict(facecolor='black', shrink=0.1)

            )

plt.axis([1, 8.5, 0, 1300])

save_fig("inertia_vs_k_diagram")

plt.show()
plot_decision_boundaries(kmeans_per_k[4-1], X)

plt.show()
from sklearn.metrics import silhouette_score
silhouette_score(X, kmeans.labels_)
silhouette_scores = [silhouette_score(X, model.labels_)

                     for model in kmeans_per_k[1:]]
plt.figure(figsize=(8, 3))

plt.plot(range(2, 10), silhouette_scores, "bo-")

plt.xlabel("$k$", fontsize=14)

plt.ylabel("Silhouette score", fontsize=14)

plt.axis([1.8, 8.5, 0.55, 0.7])

save_fig("silhouette_score_vs_k_diagram")

plt.show()
from sklearn.metrics import silhouette_samples

from matplotlib.ticker import FixedLocator, FixedFormatter



plt.figure(figsize=(11, 9))



for k in (3, 4, 5, 6):

    plt.subplot(2, 2, k - 2)

    

    y_pred = kmeans_per_k[k - 1].labels_

    silhouette_coefficients = silhouette_samples(X, y_pred)



    padding = len(X) // 30

    pos = padding

    ticks = []

    for i in range(k):

        coeffs = silhouette_coefficients[y_pred == i]

        coeffs.sort()



        color = mpl.cm.Spectral(i / k)

        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,

                          facecolor=color, edgecolor=color, alpha=0.7)

        ticks.append(pos + len(coeffs) // 2)

        pos += len(coeffs) + padding



    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))

    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))

    if k in (3, 5):

        plt.ylabel("Cluster")

    

    if k in (5, 6):

        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.xlabel("Silhouette Coefficient")

    else:

        plt.tick_params(labelbottom=False)



    plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")

    plt.title("$k={}$".format(k), fontsize=16)



save_fig("silhouette_analysis_diagram")

plt.show()
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)

X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))

X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)

X2 = X2 + [6, -8]

X = np.r_[X1, X2]

y = np.r_[y1, y2]
plot_clusters(X)
kmeans_good = KMeans(n_clusters=3, init=np.array([[-1.5, 2.5], [0.5, 0], [4, 0]]), n_init=1, random_state=42)

kmeans_bad = KMeans(n_clusters=3, random_state=42)

kmeans_good.fit(X)

kmeans_bad.fit(X)
plt.figure(figsize=(10, 3.2))



plt.subplot(121)

plot_decision_boundaries(kmeans_good, X)

plt.title("Inertia = {:.1f}".format(kmeans_good.inertia_), fontsize=14)



plt.subplot(122)

plot_decision_boundaries(kmeans_bad, X, show_ylabels=False)

plt.title("Inertia = {:.1f}".format(kmeans_bad.inertia_), fontsize=14)



save_fig("bad_kmeans_diagram")

plt.show()
# from matplotlib.image import imread

# image = imread(os.path.join("images","unsupervised_learning","ladybug.png"))

# image.shape
# X = image.reshape(-1, 3)

# kmeans = KMeans(n_clusters=8, random_state=42).fit(X)

# segmented_img = kmeans.cluster_centers_[kmeans.labels_]

# segmented_img = segmented_img.reshape(image.shape)
# segmented_imgs = []

# n_colors = (10, 8, 6, 4, 2)

# for n_clusters in n_colors:

#     kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)

#     segmented_img = kmeans.cluster_centers_[kmeans.labels_]

#     segmented_imgs.append(segmented_img.reshape(image.shape))
# plt.figure(figsize=(10,5))

# plt.subplots_adjust(wspace=0.05, hspace=0.1)



# plt.subplot(231)

# plt.imshow(image)

# plt.title("Original image")

# plt.axis('off')



# for idx, n_clusters in enumerate(n_colors):

#     plt.subplot(232 + idx)

#     plt.imshow(segmented_imgs[idx])

#     plt.title("{} colors".format(n_clusters))

#     plt.axis('off')



# save_fig('image_segmentation_diagram', tight_layout=False)

# plt.show()
from sklearn.datasets import load_digits
X_digits, y_digits = load_digits(return_X_y=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)

log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)
from sklearn.pipeline import Pipeline
pipeline = Pipeline([

    ("kmeans", KMeans(n_clusters=50, random_state=42)),

    ("log_reg", LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)),

])

pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
1 - (1 - 0.9822222) / (1 - 0.9666666)
from sklearn.model_selection import GridSearchCV
param_grid = dict(kmeans__n_clusters=range(2, 100))

grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)

grid_clf.fit(X_train, y_train)
grid_clf.best_params_
grid_clf.score(X_test, y_test)
n_labeled = 50
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)

log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])

log_reg.score(X_test, y_test)
k = 50
kmeans = KMeans(n_clusters=k, random_state=42)

X_digits_dist = kmeans.fit_transform(X_train)

representative_digit_idx = np.argmin(X_digits_dist, axis=0)

X_representative_digits = X_train[representative_digit_idx]
plt.figure(figsize=(8, 2))

for index, X_representative_digit in enumerate(X_representative_digits):

    plt.subplot(k // 10, 10, index + 1)

    plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")

    plt.axis('off')



save_fig("representative_images_diagram", tight_layout=False)

plt.show()
y_representative_digits = np.array([

    4, 8, 0, 6, 8, 3, 7, 7, 9, 2,

    5, 5, 8, 5, 2, 1, 2, 9, 6, 1,

    1, 6, 9, 0, 8, 3, 0, 7, 4, 1,

    6, 5, 2, 4, 1, 8, 6, 3, 9, 2,

    4, 2, 9, 4, 7, 6, 2, 3, 1, 1])
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)

log_reg.fit(X_representative_digits, y_representative_digits)

log_reg.score(X_test, y_test)
y_train_propagated = np.empty(len(X_train), dtype=np.int32)

for i in range(k):

    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)

log_reg.fit(X_train, y_train_propagated)
log_reg.score(X_test, y_test)
percentile_closest = 20



X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]

for i in range(k):

    in_cluster = (kmeans.labels_ == i)

    cluster_dist = X_cluster_dist[in_cluster]

    cutoff_distance = np.percentile(cluster_dist, percentile_closest)

    above_cutoff = (X_cluster_dist > cutoff_distance)

    X_cluster_dist[in_cluster & above_cutoff] = -1
partially_propagated = (X_cluster_dist != -1)

X_train_partially_propagated = X_train[partially_propagated]

y_train_partially_propagated = y_train_propagated[partially_propagated]
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)

log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
log_reg.score(X_test, y_test)
np.mean(y_train_partially_propagated == y_train[partially_propagated])
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.05, min_samples=5)

dbscan.fit(X)
dbscan.labels_[:10]
len(dbscan.core_sample_indices_)
dbscan.core_sample_indices_[:10]
dbscan.components_[:3]
np.unique(dbscan.labels_)
dbscan2 = DBSCAN(eps=0.2)

dbscan2.fit(X)
def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):

    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)

    core_mask[dbscan.core_sample_indices_] = True

    anomalies_mask = dbscan.labels_ == -1

    non_core_mask = ~(core_mask | anomalies_mask)



    cores = dbscan.components_

    anomalies = X[anomalies_mask]

    non_cores = X[non_core_mask]

    

    plt.scatter(cores[:, 0], cores[:, 1],

                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")

    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])

    plt.scatter(anomalies[:, 0], anomalies[:, 1],

                c="r", marker="x", s=100)

    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")

    if show_xlabels:

        plt.xlabel("$x_1$", fontsize=14)

    else:

        plt.tick_params(labelbottom=False)

    if show_ylabels:

        plt.ylabel("$x_2$", fontsize=14, rotation=0)

    else:

        plt.tick_params(labelleft=False)

    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)
plt.figure(figsize=(9, 3.2))



plt.subplot(121)

plot_dbscan(dbscan, X, size=100)



plt.subplot(122)

plot_dbscan(dbscan2, X, size=600, show_ylabels=False)



save_fig("dbscan_diagram")

plt.show()

dbscan = dbscan2
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=50)

knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])

knn.predict(X_new)
knn.predict_proba(X_new)
plt.figure(figsize=(6, 3))

plot_decision_boundaries(knn, X, show_centroids=False)

plt.scatter(X_new[:, 0], X_new[:, 1], c="b", marker="+", s=200, zorder=10)

save_fig("cluster_classification_diagram")

plt.show()
y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)

y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]

y_pred[y_dist > 0.2] = -1

y_pred.ravel()
from sklearn.cluster import SpectralClustering
sc1 = SpectralClustering(n_clusters=2, gamma=100, random_state=42)

sc1.fit(X)
sc2 = SpectralClustering(n_clusters=2, gamma=1, random_state=42)

sc2.fit(X)
np.percentile(sc1.affinity_matrix_, 95)
def plot_spectral_clustering(sc, X, size, alpha, show_xlabels=True, show_ylabels=True):

    plt.scatter(X[:, 0], X[:, 1], marker='o', s=size, c='gray', cmap="Paired", alpha=alpha)

    plt.scatter(X[:, 0], X[:, 1], marker='o', s=30, c='w')

    plt.scatter(X[:, 0], X[:, 1], marker='.', s=10, c=sc.labels_, cmap="Paired")

    

    if show_xlabels:

        plt.xlabel("$x_1$", fontsize=14)

    else:

        plt.tick_params(labelbottom=False)

    if show_ylabels:

        plt.ylabel("$x_2$", fontsize=14, rotation=0)

    else:

        plt.tick_params(labelleft=False)

    plt.title("RBF gamma={}".format(sc.gamma), fontsize=14)
plt.figure(figsize=(9, 3.2))



plt.subplot(121)

plot_spectral_clustering(sc1, X, size=500, alpha=0.1)



plt.subplot(122)

plot_spectral_clustering(sc2, X, size=4000, alpha=0.01, show_ylabels=False)



plt.show()

from sklearn.cluster import AgglomerativeClustering
X = np.array([0, 2, 5, 8.5]).reshape(-1, 1)

agg = AgglomerativeClustering(linkage="complete").fit(X)
learned_parameters(agg)
agg.children_
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)

X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))

X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)

X2 = X2 + [6, -8]

X = np.r_[X1, X2]

y = np.r_[y1, y2]
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=3, n_init=10, random_state=42)

gm.fit(X)
gm.weights_
gm.means_
gm.covariances_
gm.converged_
gm.n_iter_
gm.predict(X)
gm.predict_proba(X)
X_new, y_new = gm.sample(6)

X_new
y_new
gm.score_samples(X)
resolution = 100

grid = np.arange(-10, 10, 1 / resolution)

xx, yy = np.meshgrid(grid, grid)

X_full = np.vstack([xx.ravel(), yy.ravel()]).T



pdf = np.exp(gm.score_samples(X_full))

pdf_probas = pdf * (1 / resolution) ** 2

pdf_probas.sum()
from matplotlib.colors import LogNorm



def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):

    mins = X.min(axis=0) - 0.1

    maxs = X.max(axis=0) + 0.1

    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),

                         np.linspace(mins[1], maxs[1], resolution))

    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)



    plt.contourf(xx, yy, Z,

                 norm=LogNorm(vmin=1.0, vmax=30.0),

                 levels=np.logspace(0, 2, 12))

    plt.contour(xx, yy, Z,

                norm=LogNorm(vmin=1.0, vmax=30.0),

                levels=np.logspace(0, 2, 12),

                linewidths=1, colors='k')



    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z,

                linewidths=2, colors='r', linestyles='dashed')

    

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

    plot_centroids(clusterer.means_, clusterer.weights_)



    plt.xlabel("$x_1$", fontsize=14)

    if show_ylabels:

        plt.ylabel("$x_2$", fontsize=14, rotation=0)

    else:

        plt.tick_params(labelleft=False)
plt.figure(figsize=(8, 4))



plot_gaussian_mixture(gm, X)



save_fig("gaussian_mixtures_diagram")

plt.show()
gm_full = GaussianMixture(n_components=3, n_init=10, covariance_type="full", random_state=42)

gm_tied = GaussianMixture(n_components=3, n_init=10, covariance_type="tied", random_state=42)

gm_spherical = GaussianMixture(n_components=3, n_init=10, covariance_type="spherical", random_state=42)

gm_diag = GaussianMixture(n_components=3, n_init=10, covariance_type="diag", random_state=42)

gm_full.fit(X)

gm_tied.fit(X)

gm_spherical.fit(X)

gm_diag.fit(X)
def compare_gaussian_mixtures(gm1, gm2, X):

    plt.figure(figsize=(9, 4))



    plt.subplot(121)

    plot_gaussian_mixture(gm1, X)

    plt.title('covariance_type="{}"'.format(gm1.covariance_type), fontsize=14)



    plt.subplot(122)

    plot_gaussian_mixture(gm2, X, show_ylabels=False)

    plt.title('covariance_type="{}"'.format(gm2.covariance_type), fontsize=14)

compare_gaussian_mixtures(gm_tied, gm_spherical, X)



save_fig("covariance_type_diagram")

plt.show()
compare_gaussian_mixtures(gm_full, gm_diag, X)

plt.tight_layout()

plt.show()
densities = gm.score_samples(X)

density_threshold = np.percentile(densities, 4)

anomalies = X[densities < density_threshold]
plt.figure(figsize=(8, 4))



plot_gaussian_mixture(gm, X)

plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', marker='*')

plt.ylim(top=5.1)



save_fig("mixture_anomaly_detection_diagram")

plt.show()
gm.bic(X)
gm.aic(X)
n_clusters = 3

n_dims = 2

n_params_for_weights = n_clusters - 1

n_params_for_means = n_clusters * n_dims

n_params_for_covariance = n_clusters * n_dims * (n_dims + 1) // 2

n_params = n_params_for_weights + n_params_for_means + n_params_for_covariance

max_log_likelihood = gm.score(X) * len(X) # log(L^)

bic = np.log(len(X)) * n_params - 2 * max_log_likelihood

aic = 2 * n_params - 2 * max_log_likelihood
bic, aic
n_params
gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X)

             for k in range(1, 11)]
bics = [model.bic(X) for model in gms_per_k]

aics = [model.aic(X) for model in gms_per_k]
plt.figure(figsize=(8, 3))

plt.plot(range(1, 11), bics, "bo-", label="BIC")

plt.plot(range(1, 11), aics, "go--", label="AIC")

plt.xlabel("$k$", fontsize=14)

plt.ylabel("Information Criterion", fontsize=14)

plt.axis([1, 9.5, np.min(aics) - 50, np.max(aics) + 50])

plt.annotate('Minimum',

             xy=(3, bics[2]),

             xytext=(0.35, 0.6),

             textcoords='figure fraction',

             fontsize=14,

             arrowprops=dict(facecolor='black', shrink=0.1)

            )

plt.legend()

save_fig("aic_bic_vs_k_diagram")

plt.show()
min_bic = np.infty



for k in range(1, 11):

    for covariance_type in ("full", "tied", "spherical", "diag"):

        bic = GaussianMixture(n_components=k, n_init=10,

                              covariance_type=covariance_type,

                              random_state=42).fit(X).bic(X)

        if bic < min_bic:

            min_bic = bic

            best_k = k

            best_covariance_type = covariance_type
best_k
best_covariance_type
from sklearn.mixture import BayesianGaussianMixture
bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)

bgm.fit(X)
np.round(bgm.weights_, 2)
plt.figure(figsize=(8, 5))

plot_gaussian_mixture(bgm, X)

plt.show()
bgm_low = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,

                                  weight_concentration_prior=0.01, random_state=42)

bgm_high = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,

                                  weight_concentration_prior=10000, random_state=42)

nn = 73

bgm_low.fit(X[:nn])

bgm_high.fit(X[:nn])
np.round(bgm_low.weights_, 2)
np.round(bgm_high.weights_, 2)
plt.figure(figsize=(9, 4))



plt.subplot(121)

plot_gaussian_mixture(bgm_low, X[:nn])

plt.title("weight_concentration_prior = 0.01", fontsize=14)



plt.subplot(122)

plot_gaussian_mixture(bgm_high, X[:nn], show_ylabels=False)

plt.title("weight_concentration_prior = 10000", fontsize=14)



save_fig("mixture_concentration_prior_diagram")

plt.show()
X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)
bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)

bgm.fit(X_moons)
plt.figure(figsize=(9, 3.2))



plt.subplot(121)

plot_data(X_moons)

plt.xlabel("$x_1$", fontsize=14)

plt.ylabel("$x_2$", fontsize=14, rotation=0)



plt.subplot(122)

plot_gaussian_mixture(bgm, X_moons, show_ylabels=False)



save_fig("moons_vs_bgm_diagram")

plt.show()
from scipy.stats import norm
xx = np.linspace(-6, 4, 101)

ss = np.linspace(1, 2, 101)

XX, SS = np.meshgrid(xx, ss)

ZZ = 2 * norm.pdf(XX - 1.0, 0, SS) + norm.pdf(XX + 4.0, 0, SS)

ZZ = ZZ / ZZ.sum(axis=1) / (xx[1] - xx[0])
from matplotlib.patches import Polygon



plt.figure(figsize=(8, 4.5))



x_idx = 85

s_idx = 30



plt.subplot(221)

plt.contourf(XX, SS, ZZ, cmap="GnBu")

plt.plot([-6, 4], [ss[s_idx], ss[s_idx]], "k-", linewidth=2)

plt.plot([xx[x_idx], xx[x_idx]], [1, 2], "b-", linewidth=2)

plt.xlabel(r"$x$")

plt.ylabel(r"$\theta$", fontsize=14, rotation=0)

plt.title(r"Model $f(x; \theta)$", fontsize=14)



plt.subplot(222)

plt.plot(ss, ZZ[:, x_idx], "b-")

max_idx = np.argmax(ZZ[:, x_idx])

max_val = np.max(ZZ[:, x_idx])

plt.plot(ss[max_idx], max_val, "r.")

plt.plot([ss[max_idx], ss[max_idx]], [0, max_val], "r:")

plt.plot([0, ss[max_idx]], [max_val, max_val], "r:")

plt.text(1.01, max_val + 0.005, r"$\hat{L}$", fontsize=14)

plt.text(ss[max_idx]+ 0.01, 0.055, r"$\hat{\theta}$", fontsize=14)

plt.text(ss[max_idx]+ 0.01, max_val - 0.012, r"$Max$", fontsize=12)

plt.axis([1, 2, 0.05, 0.15])

plt.xlabel(r"$\theta$", fontsize=14)

plt.grid(True)

plt.text(1.99, 0.135, r"$=f(x=2.5; \theta)$", fontsize=14, ha="right")

plt.title(r"Likelihood function $\mathcal{L}(\theta|x=2.5)$", fontsize=14)



plt.subplot(223)

plt.plot(xx, ZZ[s_idx], "k-")

plt.axis([-6, 4, 0, 0.25])

plt.xlabel(r"$x$", fontsize=14)

plt.grid(True)

plt.title(r"PDF $f(x; \theta=1.3)$", fontsize=14)

verts = [(xx[41], 0)] + list(zip(xx[41:81], ZZ[s_idx, 41:81])) + [(xx[80], 0)]

poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')

plt.gca().add_patch(poly)



plt.subplot(224)

plt.plot(ss, np.log(ZZ[:, x_idx]), "b-")

max_idx = np.argmax(np.log(ZZ[:, x_idx]))

max_val = np.max(np.log(ZZ[:, x_idx]))

plt.plot(ss[max_idx], max_val, "r.")

plt.plot([ss[max_idx], ss[max_idx]], [-5, max_val], "r:")

plt.plot([0, ss[max_idx]], [max_val, max_val], "r:")

plt.axis([1, 2, -2.4, -2])

plt.xlabel(r"$\theta$", fontsize=14)

plt.text(ss[max_idx]+ 0.01, max_val - 0.05, r"$Max$", fontsize=12)

plt.text(ss[max_idx]+ 0.01, -2.39, r"$\hat{\theta}$", fontsize=14)

plt.text(1.01, max_val + 0.02, r"$\log \, \hat{L}$", fontsize=14)

plt.grid(True)

plt.title(r"$\log \, \mathcal{L}(\theta|x=2.5)$", fontsize=14)



save_fig("likelihood_function_diagram")

plt.show()
X_train = mnist['data'][:60000]

y_train = mnist['target'][:60000]



X_test = mnist['data'][60000:]

y_test = mnist['target'][60000:]
from sklearn.ensemble import RandomForestClassifier



rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
import time



t0 = time.time()

rnd_clf.fit(X_train, y_train)

t1 = time.time()
print("Training took {:.2f}s".format(t1 - t0))
from sklearn.metrics import accuracy_score



y_pred = rnd_clf.predict(X_test)

accuracy_score(y_test, y_pred)
from sklearn.decomposition import PCA



pca = PCA(n_components=0.95)

X_train_reduced = pca.fit_transform(X_train)
rnd_clf2 = RandomForestClassifier(n_estimators=10, random_state=42)

t0 = time.time()

rnd_clf2.fit(X_train_reduced, y_train)

t1 = time.time()
print("Training took {:.2f}s".format(t1 - t0))
X_test_reduced = pca.transform(X_test)



y_pred = rnd_clf2.predict(X_test_reduced)

accuracy_score(y_test, y_pred)
from sklearn.linear_model import LogisticRegression



log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)

t0 = time.time()

log_clf.fit(X_train, y_train)

t1 = time.time()
print("Training took {:.2f}s".format(t1 - t0))
y_pred = log_clf.predict(X_test)

accuracy_score(y_test, y_pred)
log_clf2 = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)

t0 = time.time()

log_clf2.fit(X_train_reduced, y_train)

t1 = time.time()
print("Training took {:.2f}s".format(t1 - t0))
y_pred = log_clf2.predict(X_test_reduced)

accuracy_score(y_test, y_pred)
np.random.seed(42)



m = 10000

idx = np.random.permutation(60000)[:m]



X = mnist['data'][idx]

y = mnist['target'][idx]
from sklearn.manifold import TSNE



tsne = TSNE(n_components=2, random_state=42)

X_reduced = tsne.fit_transform(X)
plt.figure(figsize=(13,10))

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="jet")

plt.axis('off')

plt.colorbar()

plt.show()
plt.figure(figsize=(9,9))

cmap = mpl.cm.get_cmap("jet")

for digit in (2, 3, 5):

    plt.scatter(X_reduced[y == digit, 0], X_reduced[y == digit, 1], c=[cmap(digit / 9)])

plt.axis('off')

plt.show()
idx = (y == 2) | (y == 3) | (y == 5) 

X_subset = X[idx]

y_subset = y[idx]



tsne_subset = TSNE(n_components=2, random_state=42)

X_subset_reduced = tsne_subset.fit_transform(X_subset)
plt.figure(figsize=(9,9))

for digit in (2, 3, 5):

    plt.scatter(X_subset_reduced[y_subset == digit, 0], X_subset_reduced[y_subset == digit, 1], c=[cmap(digit / 9)])

plt.axis('off')

plt.show()
from sklearn.preprocessing import MinMaxScaler

from matplotlib.offsetbox import AnnotationBbox, OffsetImage



def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13, 10)):

    # Let's scale the input features so that they range from 0 to 1

    X_normalized = MinMaxScaler().fit_transform(X)

    # Now we create the list of coordinates of the digits plotted so far.

    # We pretend that one is already plotted far away at the start, to

    # avoid `if` statements in the loop below

    neighbors = np.array([[10., 10.]])

    # The rest should be self-explanatory

    plt.figure(figsize=figsize)

    cmap = mpl.cm.get_cmap("jet")

    digits = np.unique(y)

    for digit in digits:

        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=[cmap(digit / 9)])

    plt.axis("off")

    ax = plt.gcf().gca()  # get current axes in current figure

    for index, image_coord in enumerate(X_normalized):

        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()

        if closest_distance > min_distance:

            neighbors = np.r_[neighbors, [image_coord]]

            if images is None:

                plt.text(image_coord[0], image_coord[1], str(int(y[index])),

                         color=cmap(y[index] / 9), fontdict={"weight": "bold", "size": 16})

            else:

                image = images[index].reshape(28, 28)

                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)

                ax.add_artist(imagebox)
plot_digits(X_reduced, y)
plot_digits(X_reduced, y, images=X, figsize=(35, 25))
plot_digits(X_subset_reduced, y_subset, images=X_subset, figsize=(22, 22))
from sklearn.decomposition import PCA

import time



t0 = time.time()

X_pca_reduced = PCA(n_components=2, random_state=42).fit_transform(X)

t1 = time.time()

print("PCA took {:.1f}s.".format(t1 - t0))

plot_digits(X_pca_reduced, y)

plt.show()
from sklearn.manifold import LocallyLinearEmbedding



t0 = time.time()

X_lle_reduced = LocallyLinearEmbedding(n_components=2, random_state=42).fit_transform(X)

t1 = time.time()

print("LLE took {:.1f}s.".format(t1 - t0))

plot_digits(X_lle_reduced, y)

plt.show()
from sklearn.pipeline import Pipeline



pca_lle = Pipeline([

    ("pca", PCA(n_components=0.95, random_state=42)),

    ("lle", LocallyLinearEmbedding(n_components=2, random_state=42)),

])

t0 = time.time()

X_pca_lle_reduced = pca_lle.fit_transform(X)

t1 = time.time()

print("PCA+LLE took {:.1f}s.".format(t1 - t0))

plot_digits(X_pca_lle_reduced, y)

plt.show()
from sklearn.manifold import MDS



m = 2000

t0 = time.time()

X_mds_reduced = MDS(n_components=2, random_state=42).fit_transform(X[:m])

t1 = time.time()

print("MDS took {:.1f}s (on just 2,000 MNIST images instead of 10,000).".format(t1 - t0))

plot_digits(X_mds_reduced, y[:m])

plt.show()
from sklearn.pipeline import Pipeline



pca_mds = Pipeline([

    ("pca", PCA(n_components=0.95, random_state=42)),

    ("mds", MDS(n_components=2, random_state=42)),

])

t0 = time.time()

X_pca_mds_reduced = pca_mds.fit_transform(X[:2000])

t1 = time.time()

print("PCA+MDS took {:.1f}s (on 2,000 MNIST images).".format(t1 - t0))

plot_digits(X_pca_mds_reduced, y[:2000])

plt.show()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



t0 = time.time()

X_lda_reduced = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)

t1 = time.time()

print("LDA took {:.1f}s.".format(t1 - t0))

plot_digits(X_lda_reduced, y, figsize=(12,12))

plt.show()
from sklearn.manifold import TSNE



t0 = time.time()

X_tsne_reduced = TSNE(n_components=2, random_state=42).fit_transform(X)

t1 = time.time()

print("t-SNE took {:.1f}s.".format(t1 - t0))

plot_digits(X_tsne_reduced, y)

plt.show()
pca_tsne = Pipeline([

    ("pca", PCA(n_components=0.95, random_state=42)),

    ("tsne", TSNE(n_components=2, random_state=42)),

])

t0 = time.time()

X_pca_tsne_reduced = pca_tsne.fit_transform(X)

t1 = time.time()

print("PCA+t-SNE took {:.1f}s.".format(t1 - t0))

plot_digits(X_pca_tsne_reduced, y)

plt.show()