# Python ≥3.5 is required

import sys



# Common imports

import numpy as np

import os



import warnings

warnings.filterwarnings('ignore')

from sklearn.svm import SVC

from sklearn import datasets

import numpy as np

from sklearn import datasets

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC



from sklearn.datasets import make_moons

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures





# to make this notebook's output stable across runs

np.random.seed(42)



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)
iris = datasets.load_iris()

X = iris["data"][:, (2, 3)]  # petal length, petal width

y = iris["target"]
setosa_or_versicolor = (y == 0) | (y == 1)

X = X[setosa_or_versicolor]

y = y[setosa_or_versicolor]

# SVM Classifier model

svm_clf = SVC(kernel="linear", C=float("inf"))

svm_clf.fit(X, y)
# Bad models

x0 = np.linspace(0, 5.5, 200)

pred_1 = 5*x0 - 20

pred_2 = x0 - 1.8

pred_3 = 0.1 * x0 + 0.5


def plot_svc_decision_boundary(svm_clf, xmin, xmax):

    w = svm_clf.coef_[0]

    b = svm_clf.intercept_[0]



    # At the decision boundary, w0*x0 + w1*x1 + b = 0

    # => x1 = -w0/w1 * x0 - b/w1

    x0 = np.linspace(xmin, xmax, 200)

    decision_boundary = -w[0]/w[1] * x0 - b/w[1]



    margin = 1/w[1]

    gutter_up = decision_boundary + margin

    gutter_down = decision_boundary - margin



    svs = svm_clf.support_vectors_

    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')

    plt.plot(x0, decision_boundary, "k-", linewidth=2)

    plt.plot(x0, gutter_up, "k--", linewidth=2)

    plt.plot(x0, gutter_down, "k--", linewidth=2)
fig, axes = plt.subplots(ncols=2, figsize=(10,2.7), sharey=True)



plt.sca(axes[0])

plt.plot(x0, pred_1, "g--", linewidth=2)

plt.plot(x0, pred_2, "m-", linewidth=2)

plt.plot(x0, pred_3, "r-", linewidth=2)

plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")

plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")

plt.xlabel("Petal length", fontsize=14)

plt.ylabel("Petal width", fontsize=14)

plt.legend(loc="upper left", fontsize=14)

plt.axis([0, 5.5, 0, 2])



plt.sca(axes[1])

plot_svc_decision_boundary(svm_clf, 0, 5.5)

plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")

plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo")

plt.xlabel("Petal length", fontsize=14)

plt.axis([0, 5.5, 0, 2])

plt.show()
Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)

ys = np.array([0, 0, 1, 1])

svm_clf = SVC(kernel="linear", C=100)

svm_clf.fit(Xs, ys)



plt.figure(figsize=(9,2.7))

plt.subplot(121)

plt.plot(Xs[:, 0][ys==1], Xs[:, 1][ys==1], "bo")

plt.plot(Xs[:, 0][ys==0], Xs[:, 1][ys==0], "ms")

plot_svc_decision_boundary(svm_clf, 0, 6)

plt.xlabel("$x_0$", fontsize=20)

plt.ylabel("$x_1$    ", fontsize=20, rotation=0)

plt.title("Unscaled", fontsize=16)

plt.axis([0, 6, 0, 90])



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(Xs)

svm_clf.fit(X_scaled, ys)



plt.subplot(122)

plt.plot(X_scaled[:, 0][ys==1], X_scaled[:, 1][ys==1], "bo")

plt.plot(X_scaled[:, 0][ys==0], X_scaled[:, 1][ys==0], "ms")

plot_svc_decision_boundary(svm_clf, -2, 2)

plt.xlabel("$x_0$", fontsize=20)

plt.ylabel("$x'_1$  ", fontsize=20, rotation=0)

plt.title("Scaled", fontsize=16)

plt.axis([-2, 2, -2, 2])

plt.show()
X_outliers = np.array([[3.4, 1.3], [3.2, 0.8]])

y_outliers = np.array([0, 0])

Xo1 = np.concatenate([X, X_outliers[:1]], axis=0)

yo1 = np.concatenate([y, y_outliers[:1]], axis=0)

Xo2 = np.concatenate([X, X_outliers[1:]], axis=0)

yo2 = np.concatenate([y, y_outliers[1:]], axis=0)



svm_clf2 = SVC(kernel="linear", C=10**9)

svm_clf2.fit(Xo2, yo2)



fig, axes = plt.subplots(ncols=2, figsize=(10,2.7), sharey=True)



plt.sca(axes[0])

plt.plot(Xo1[:, 0][yo1==1], Xo1[:, 1][yo1==1], "bs")

plt.plot(Xo1[:, 0][yo1==0], Xo1[:, 1][yo1==0], "yo")

plt.text(0.3, 1.0, "Impossible!", fontsize=24, color="red")

plt.xlabel("Petal length", fontsize=14)

plt.ylabel("Petal width", fontsize=14)

plt.annotate("Outlier",

             xy=(X_outliers[0][0], X_outliers[0][1]),

             xytext=(2.5, 1.7),

             ha="center",

             arrowprops=dict(facecolor='black', shrink=0.1),

             fontsize=16,

            )

plt.axis([0, 5.5, 0, 2])



plt.sca(axes[1])

plt.plot(Xo2[:, 0][yo2==1], Xo2[:, 1][yo2==1], "bs")

plt.plot(Xo2[:, 0][yo2==0], Xo2[:, 1][yo2==0], "yo")

plot_svc_decision_boundary(svm_clf2, 0, 5.5)

plt.xlabel("Petal length", fontsize=14)

plt.annotate("Outlier",

             xy=(X_outliers[1][0], X_outliers[1][1]),

             xytext=(3.2, 0.08),

             ha="center",

             arrowprops=dict(facecolor='black', shrink=0.1),

             fontsize=16,

            )

plt.axis([0, 5.5, 0, 2])

plt.show()
iris = datasets.load_iris()

X = iris["data"][:, (2, 3)]  # petal length, petal width

y = (iris["target"] == 2).astype(np.float64)  # Iris virginica



svm_clf = Pipeline([

        ("scaler", StandardScaler()),

        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),

    ])



svm_clf.fit(X, y)
svm_clf.predict([[5.5, 1.7]])
scaler = StandardScaler()

svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)

svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)



scaled_svm_clf1 = Pipeline([

        ("scaler", scaler),

        ("linear_svc", svm_clf1),

    ])

scaled_svm_clf2 = Pipeline([

        ("scaler", scaler),

        ("linear_svc", svm_clf2),

    ])



scaled_svm_clf1.fit(X, y)

scaled_svm_clf2.fit(X, y)
# Convert to unscaled parameters

b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])

b2 = svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])

w1 = svm_clf1.coef_[0] / scaler.scale_

w2 = svm_clf2.coef_[0] / scaler.scale_

svm_clf1.intercept_ = np.array([b1])

svm_clf2.intercept_ = np.array([b2])

svm_clf1.coef_ = np.array([w1])

svm_clf2.coef_ = np.array([w2])



# Find support vectors (LinearSVC does not do this automatically)

t = y * 2 - 1

support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()

support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()

svm_clf1.support_vectors_ = X[support_vectors_idx1]

svm_clf2.support_vectors_ = X[support_vectors_idx2]
fig, axes = plt.subplots(ncols=2, figsize=(10,2.7), sharey=True)



plt.sca(axes[0])

plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Iris virginica")

plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Iris versicolor")

plot_svc_decision_boundary(svm_clf1, 4, 5.9)

plt.xlabel("Petal length", fontsize=14)

plt.ylabel("Petal width", fontsize=14)

plt.legend(loc="upper left", fontsize=14)

plt.title("$C = {}$".format(svm_clf1.C), fontsize=16)

plt.axis([4, 5.9, 0.8, 2.8])



plt.sca(axes[1])

plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")

plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")

plot_svc_decision_boundary(svm_clf2, 4, 5.99)

plt.xlabel("Petal length", fontsize=14)

plt.title("$C = {}$".format(svm_clf2.C), fontsize=16)

plt.axis([4, 5.9, 0.8, 2.8])

plt.show()
X1D = np.linspace(-4, 4, 9).reshape(-1, 1)

X2D = np.c_[X1D, X1D**2]

y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])



plt.figure(figsize=(10, 3))



plt.subplot(121)

plt.grid(True, which='both')

plt.axhline(y=0, color='k')

plt.plot(X1D[:, 0][y==0], np.zeros(4), "bs")

plt.plot(X1D[:, 0][y==1], np.zeros(5), "g^")

plt.gca().get_yaxis().set_ticks([])

plt.xlabel(r"$x_1$", fontsize=20)

plt.axis([-4.5, 4.5, -0.2, 0.2])



plt.subplot(122)

plt.grid(True, which='both')

plt.axhline(y=0, color='k')

plt.axvline(x=0, color='k')

plt.plot(X2D[:, 0][y==0], X2D[:, 1][y==0], "bs")

plt.plot(X2D[:, 0][y==1], X2D[:, 1][y==1], "g^")

plt.xlabel(r"$x_1$", fontsize=20)

plt.ylabel(r"$x_2$  ", fontsize=20, rotation=0)

plt.gca().get_yaxis().set_ticks([0, 4, 8, 12, 16])

plt.plot([-4.5, 4.5], [6.5, 6.5], "r--", linewidth=3)

plt.axis([-4.5, 4.5, -1, 17])



plt.subplots_adjust(right=1)



plt.show()
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)



def plot_dataset(X, y, axes):

    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")

    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")

    plt.axis(axes)

    plt.grid(True, which='both')

    plt.xlabel(r"$x_1$", fontsize=20)

    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)



plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])

plt.show()
polynomial_svm_clf = Pipeline([

        ("poly_features", PolynomialFeatures(degree=3)),

        ("scaler", StandardScaler()),

        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))

    ])



polynomial_svm_clf.fit(X, y)
def plot_predictions(clf, axes):

    x0s = np.linspace(axes[0], axes[1], 100)

    x1s = np.linspace(axes[2], axes[3], 100)

    x0, x1 = np.meshgrid(x0s, x1s)

    X = np.c_[x0.ravel(), x1.ravel()]

    y_pred = clf.predict(X).reshape(x0.shape)

    y_decision = clf.decision_function(X).reshape(x0.shape)

    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)

    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)



plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])

plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])



plt.show()
def gaussian_rbf(x, landmark, gamma):

    return np.exp(-gamma * np.linalg.norm(x - landmark, axis=1)**2)



gamma = 0.3



x1s = np.linspace(-4.5, 4.5, 200).reshape(-1, 1)

x2s = gaussian_rbf(x1s, -2, gamma)

x3s = gaussian_rbf(x1s, 1, gamma)



XK = np.c_[gaussian_rbf(X1D, -2, gamma), gaussian_rbf(X1D, 1, gamma)]

yk = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])



plt.figure(figsize=(10.5, 4))



plt.subplot(121)

plt.grid(True, which='both')

plt.axhline(y=0, color='k')

plt.scatter(x=[-2, 1], y=[0, 0], s=150, alpha=0.5, c="red")

plt.plot(X1D[:, 0][yk==0], np.zeros(4), "bs")

plt.plot(X1D[:, 0][yk==1], np.zeros(5), "g^")

plt.plot(x1s, x2s, "g--")

plt.plot(x1s, x3s, "b:")

plt.gca().get_yaxis().set_ticks([0, 0.25, 0.5, 0.75, 1])

plt.xlabel(r"$x_1$", fontsize=20)

plt.ylabel(r"Similarity", fontsize=14)

plt.annotate(r'$\mathbf{x}$',

             xy=(X1D[3, 0], 0),

             xytext=(-0.5, 0.20),

             ha="center",

             arrowprops=dict(facecolor='black', shrink=0.1),

             fontsize=18,

            )

plt.text(-2, 0.9, "$x_2$", ha="center", fontsize=20)

plt.text(1, 0.9, "$x_3$", ha="center", fontsize=20)

plt.axis([-4.5, 4.5, -0.1, 1.1])



plt.subplot(122)

plt.grid(True, which='both')

plt.axhline(y=0, color='k')

plt.axvline(x=0, color='k')

plt.plot(XK[:, 0][yk==0], XK[:, 1][yk==0], "bs")

plt.plot(XK[:, 0][yk==1], XK[:, 1][yk==1], "g^")

plt.xlabel(r"$x_2$", fontsize=20)

plt.ylabel(r"$x_3$  ", fontsize=20, rotation=0)

plt.annotate(r'$\phi\left(\mathbf{x}\right)$',

             xy=(XK[3, 0], XK[3, 1]),

             xytext=(0.65, 0.50),

             ha="center",

             arrowprops=dict(facecolor='black', shrink=0.1),

             fontsize=18,

            )

plt.plot([-0.1, 1.1], [0.57, -0.1], "r--", linewidth=3)

plt.axis([-0.1, 1.1, -0.1, 1.1])

    

plt.subplots_adjust(right=1)

plt.show()
x1_example = X1D[3, 0]

for landmark in (-2, 1):

    k = gaussian_rbf(np.array([[x1_example]]), np.array([[landmark]]), gamma)

    print("Phi({}, {}) = {}".format(x1_example, landmark, k))
rbf_kernel_svm_clf = Pipeline([

        ("scaler", StandardScaler()),

        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))

    ])

rbf_kernel_svm_clf.fit(X, y)
from sklearn.svm import SVC



gamma1, gamma2 = 0.1, 5

C1, C2 = 0.001, 1000

hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)



svm_clfs = []

for gamma, C in hyperparams:

    rbf_kernel_svm_clf = Pipeline([

            ("scaler", StandardScaler()),

            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))

        ])

    rbf_kernel_svm_clf.fit(X, y)

    svm_clfs.append(rbf_kernel_svm_clf)



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10.5, 7), sharex=True, sharey=True)



for i, svm_clf in enumerate(svm_clfs):

    plt.sca(axes[i // 2, i % 2])

    plot_predictions(svm_clf, [-1.5, 2.45, -1, 1.5])

    plot_dataset(X, y, [-1.5, 2.45, -1, 1.5])

    gamma, C = hyperparams[i]

    plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)

    if i in (0, 1):

        plt.xlabel("")

    if i in (1, 3):

        plt.ylabel("")





plt.show()
np.random.seed(42)

m = 50

X = 2 * np.random.rand(m, 1)

y = (4 + 3 * X + np.random.randn(m, 1)).ravel()
from sklearn.svm import LinearSVR



svm_reg = LinearSVR(epsilon=1.5, random_state=42)

svm_reg.fit(X, y)
svm_reg1 = LinearSVR(epsilon=1.5, random_state=42)

svm_reg2 = LinearSVR(epsilon=0.5, random_state=42)

svm_reg1.fit(X, y)

svm_reg2.fit(X, y)



def find_support_vectors(svm_reg, X, y):

    y_pred = svm_reg.predict(X)

    off_margin = (np.abs(y - y_pred) >= svm_reg.epsilon)

    return np.argwhere(off_margin)



svm_reg1.support_ = find_support_vectors(svm_reg1, X, y)

svm_reg2.support_ = find_support_vectors(svm_reg2, X, y)



eps_x1 = 1

eps_y_pred = svm_reg1.predict([[eps_x1]])
def plot_svm_regression(svm_reg, X, y, axes):

    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)

    y_pred = svm_reg.predict(x1s)

    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")

    plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")

    plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")

    plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')

    plt.plot(X, y, "bo")

    plt.xlabel(r"$x_1$", fontsize=18)

    plt.legend(loc="upper left", fontsize=18)

    plt.axis(axes)



fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)

plt.sca(axes[0])

plot_svm_regression(svm_reg1, X, y, [0, 2, 3, 11])

plt.title(r"$\epsilon = {}$".format(svm_reg1.epsilon), fontsize=18)

plt.ylabel(r"$y$", fontsize=18, rotation=0)

#plt.plot([eps_x1, eps_x1], [eps_y_pred, eps_y_pred - svm_reg1.epsilon], "k-", linewidth=2)

plt.annotate(

        '', xy=(eps_x1, eps_y_pred), xycoords='data',

        xytext=(eps_x1, eps_y_pred - svm_reg1.epsilon),

        textcoords='data', arrowprops={'arrowstyle': '<->', 'linewidth': 1.5}

    )

plt.text(0.91, 5.6, r"$\epsilon$", fontsize=20)

plt.sca(axes[1])

plot_svm_regression(svm_reg2, X, y, [0, 2, 3, 11])

plt.title(r"$\epsilon = {}$".format(svm_reg2.epsilon), fontsize=18)



plt.show()