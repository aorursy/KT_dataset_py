import sys

import sklearn

import numpy as np

import os



np.random.seed(42)



%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")
# Some random data : Let's find the solution using the normal equation

X = 2 * np.random.rand(100,1)

y = 4 + 3*X + np.random.randn(100, 1) # y = 4 + 3X + noise
plt.plot(X, y, "b.")

plt.xlabel("$X_1$", fontsize=18)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.axis([0, 2, 0, 15])

plt.show()
# Our estimate

X_with_bias = np.c_[np.ones((100, 1)), X] # Adding ones to each instance since x0 = 1

theta_best = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y) # theta = (X_t * X)^-1 * (X_t) * y
theta_best # 4.2 and 2.7 remarkably close to 4 and 3 respectively
X_lims = np.array([[0], [2]])

X_lims_with_bias = np.c_[np.ones((2, 1)), X_lims]

y_pred = X_lims_with_bias.dot(theta_best)
y_pred
plt.plot(X_lims, y_pred, "r-", linewidth=2, label="Predictions")

plt.plot(X, y, "b.")

plt.xlabel("$X_1$", fontsize=18)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.legend(loc="upper left", fontsize=14)

plt.axis([0, 2, 0, 15])

plt.show()
# Let's do the same with sklearn this time

from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(X, y)

lin_reg.intercept_, lin_reg.coef_ # Same as before
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_with_bias, y, rcond=1e-6)

theta_best_svd
eta = 0.1 # Learning rate

num_iters = 1000

m = 100 # Number of training samples



theta = np.random.randn(2, 1) # Randomly Initialize Theta



for iteration in range(num_iters):

    gradients = (2/m)*(X_with_bias.T.dot((X_with_bias.dot(theta) - y))) # grad = (2/m)*(X_t)*(X*theta - y)

    theta = theta - eta*gradients

theta # Same as before
X_lims_with_bias.dot(theta)
theta_bgd = theta # Save for later
# Gradient descent with different learning rates

theta_path_bgd = []



def plot_gradient_descent(theta, eta, theta_path=None):

    m = len(X_with_bias)

    plt.plot(X, y, "b.")

    n_iterations = 1000

    for iteration in range(n_iterations):

        if iteration < 10:

            y_predict = X_lims_with_bias.dot(theta)

            style = "g-" if iteration > 0 else "r--"

            plt.plot(X_lims, y_predict, style)

        gradients = 2/m * X_with_bias.T.dot(X_with_bias.dot(theta) - y)

        theta = theta - eta * gradients

        if theta_path is not None:

            theta_path.append(theta)

    plt.xlabel("$x_1$", fontsize=18)

    plt.axis([0, 2, 0, 15])

    plt.title(r"$\eta = {}$".format(eta), fontsize=16)
np.random.seed(42)

theta = np.random.randn(2,1)  # random initialization



plt.figure(figsize=(10,4))

plt.subplot(131); plot_gradient_descent(theta, eta=0.02)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.subplot(132); plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)

plt.subplot(133); plot_gradient_descent(theta, eta=0.5)



plt.show()
num_epochs = 50

t0, t1 = 5, 50 # Learning Schedule Hyperparameters



def learning_schedule(t):

    return t0/(t + t1)
theta = np.random.randn(2, 1) # Randomly Initialize theta



for epoch in range(num_epochs):

    for i in range(m):

        random_idx = np.random.randint(m)

        xi = X_with_bias[random_idx : random_idx + 1] # Choose a random sample in the training set

        yi = y[random_idx : random_idx + 1]

        gradients = 2 * xi.T.dot(xi.dot(theta) - yi) # Compute Gradient on just this sample

        eta = learning_schedule(epoch * m + i) # Adjust Learning Rate

        theta = theta - eta*gradients # Update theta
theta # Same-ish as before
# Using Scikit-learn, the same thing can be computed as:

from sklearn.linear_model import SGDRegressor



sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)

sgd_reg.fit(X, y.ravel())
sgd_reg.intercept_ , sgd_reg.coef_
theta_sgd = theta # Save for later
n_iterations = 50

batch_size = 20 

theta = np.random.randn(2, 1) # Initalize theta randomly



t0, t1 = 200, 1000



t = 0

for epoch in range(n_iterations):

    shuffled_indices = np.random.permutation(m)

    X_b_shuffled = X_with_bias[shuffled_indices]

    y_shuffled = y[shuffled_indices]

    for i in range(0, m , batch_size):

        t += 1

        xi = X_b_shuffled[i : i + batch_size]

        yi = y_shuffled[i : i + batch_size]

        gradients = (2/m) * (xi.T.dot(xi.dot(theta) - yi))

        eta = learning_schedule(t)

        theta = theta - eta*gradients
theta # Same-ish
theta_mbgd = theta
def plot_grad_descent(theta_bgd, theta_sgd, theta_mbgd):

    y_pred_bgd = X_lims_with_bias.dot(theta_bgd)

    y_pred_sgd = X_lims_with_bias.dot(theta_sgd)

    y_pred_mbgd = X_lims_with_bias.dot(theta_mbgd)

    plt.plot(X_lims, y_pred_bgd, "r-", linewidth=2, label="Batch Grad Descent")

    plt.plot(X_lims, y_pred_sgd, "g-", linewidth=2, label="Stochastic Grad Descent")

    plt.plot(X_lims, y_pred_mbgd, "y-", linewidth=2, label="Mini Batch Grad Descent")

    plt.plot(X, y, "b.")

    plt.xlabel("$X_1$", fontsize=18)

    plt.ylabel("$y$", rotation=0, fontsize=18)

    plt.legend(loc="upper left", fontsize=14)

    plt.axis([0, 2, 0, 15])

    plt.show()
plot_grad_descent(theta_bgd, theta_sgd, theta_mbgd) # All seem to line up just nearly identically
m = 100

X = 6 * np.random.rand(m, 1) - 3

y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1) # y = 0.5*X^2 + X + 2 + noise
plt.plot(X, y, "b.")

plt.xlabel("$x_1$", fontsize=18)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.axis([-3, 3, 0, 10])

plt.title("Quadratic Data")

plt.show()
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly_features.fit_transform(X)

X[0]
lin_reg = LinearRegression()

lin_reg.fit(X_poly, y) # Using the transformed input feature

lin_reg.intercept_, lin_reg.coef_
# Let's see how this fits our data

X_new=np.linspace(-3, 3, 100).reshape(100, 1)

X_new_poly = poly_features.transform(X_new)

y_new = lin_reg.predict(X_new_poly)

plt.plot(X, y, "b.")

plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")

plt.xlabel("$x_1$", fontsize=18)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.legend(loc="upper left", fontsize=14)

plt.axis([-3, 3, 0, 10])

plt.title("Polynomial Regression")

plt.show()
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

def plot_poly_reg(degree):

    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)

    std_scaler = StandardScaler()

    lin_reg = LinearRegression()

    polynomial_regression = Pipeline([

            ("poly_features", polybig_features),

            ("std_scaler", std_scaler),

            ("lin_reg", lin_reg),

        ])

    polynomial_regression.fit(X, y)

    y_newbig = polynomial_regression.predict(X_new)

    plt.plot(X_new, y_newbig, "r-", label=str(degree), linewidth=2)



    plt.plot(X, y, "b.", linewidth=3)

    plt.legend(loc="upper left")

    plt.xlabel("$x_1$", fontsize=18)

    plt.ylabel("$y$", rotation=0, fontsize=18)

    plt.axis([-3, 3, 0, 10])

    plt.title("$degree ={}$".format(degree), fontsize=18)
plt.figure(figsize=(30,6))

plt.subplot(131); plot_poly_reg(1) # Underfitting

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.subplot(132); plot_poly_reg(2) # About right

plt.subplot(133); plot_poly_reg(300) # Overfitting



plt.show()
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



# How does the model performance change as the size of the training set increases

def plot_learning_curve(model, X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_errors , val_errors = [], []

    

    for m in range(1, len(X_train)):

        model.fit(X_train[:m], y_train[:m])

        y_train_predict = model.predict(X_train[:m]) # Training Predictions 

        y_val_predict = model.predict(X_test) # Val Predictions

        train_errors.append(mean_squared_error(y_train[:m], y_train_predict)) # Append errors

        val_errors.append(mean_squared_error(y_test, y_val_predict)) 

    

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")

    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")

    plt.legend(loc="upper right", fontsize=14)   # not shown in the book

    plt.xlabel("Training set size", fontsize=14) # not shown

    plt.ylabel("RMSE", fontsize=14)              # not shown
lin_reg = LinearRegression()

plot_learning_curve(lin_reg, X, y)

plt.axis([0, 80, 0, 3]) # 100 samples -> 80 train values max since 20 samples will be val

plt.title("Underfitting")

plt.show()
# Overfitting 

polynomial_regression = Pipeline([

        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),

        ("lin_reg", LinearRegression()),

    ])



plot_learning_curve(polynomial_regression, X, y)

plt.axis([0, 80, 0, 3])           

plt.title("Overfitting")  

plt.show()                        
from sklearn.linear_model import Ridge



# New Random Data

m = 20

X = 3 * np.random.rand(m, 1)

y = 1 + 0.5*X + np.random.randn(m , 1) / 1.5

X_new = np.linspace(0, 3, 100).reshape(100, 1)



# Ridge Regression

for solver in ("cholesky", "sag"):

    ridge_reg = Ridge(alpha = 1, solver=solver, random_state=42)

    ridge_reg.fit(X,y)

    print(ridge_reg.predict([[1.5]]))

    
# SGD but with l2 penalty

sgd_reg = SGDRegressor(penalty="l2", max_iter=1000, tol=1e-3, random_state=42)

sgd_reg.fit(X, y.ravel())

sgd_reg.predict([[1.5]])
def plot_model(model_class, polynomial, alphas, **model_kargs):

    for alpha, style in zip(alphas, ("b-", "g--", "r:")):

        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()

        if polynomial:

            model = Pipeline([

                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),

                    ("std_scaler", StandardScaler()),

                    ("regul_reg", model),

                ])

        model.fit(X, y)

        y_new_regul = model.predict(X_new)

        lw = 2 if alpha > 0 else 1

        plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))

    plt.plot(X, y, "b.", linewidth=3)

    plt.legend(loc="upper left", fontsize=15)

    plt.xlabel("$x_1$", fontsize=18)

    plt.axis([0, 3, 0, 4])



plt.figure(figsize=(24,8))

plt.subplot(121)

plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.subplot(122)

plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)



plt.show()
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)

lasso_reg.fit(X, y)

lasso_reg.predict([[1.5]]) # Pretty similar to Ridge in the result
plt.figure(figsize=(24, 8))

plt.subplot(121)

plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.subplot(122)

plot_model(Lasso, polynomial=True, alphas=(0, 10**-7, 1), random_state=42)



plt.show()
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

elastic_net.fit(X, y)

elastic_net.predict([[1.5]])
plt.figure(figsize=(24, 8))

plt.subplot(121)

plot_model(ElasticNet, polynomial=False, alphas=(0, 0.1, 1), l1_ratio=0.5, random_state=42)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.subplot(122)

plot_model(ElasticNet, polynomial=True, alphas=(0, 10**-7, 1), l1_ratio=0.5, random_state=42)



plt.show()
# Some fresh data

np.random.seed(42)

m = 100

X = 6 * np.random.rand(m, 1) - 3

y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)



X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)
from sklearn.base import clone



# Data Preparation Pipeline

poly_scaler = Pipeline([

        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),

        ("std_scaler", StandardScaler())

    ])



X_train_poly_scaled = poly_scaler.fit_transform(X_train)

X_val_poly_scaled = poly_scaler.transform(X_val)



sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,

                       penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)



minimum_val_error = float("inf")

best_epoch = None

best_model = None

for epoch in range(1000):

    sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off

    y_val_predict = sgd_reg.predict(X_val_poly_scaled)

    val_error = mean_squared_error(y_val, y_val_predict)

    if val_error < minimum_val_error:

        minimum_val_error = val_error

        best_epoch = epoch

        best_model = clone(sgd_reg) # Save our best model and epoch
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,

                       penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)



n_epochs = 500

train_errors, val_errors = [], []

for epoch in range(n_epochs):

    sgd_reg.fit(X_train_poly_scaled, y_train)

    y_train_predict = sgd_reg.predict(X_train_poly_scaled)

    y_val_predict = sgd_reg.predict(X_val_poly_scaled)

    train_errors.append(mean_squared_error(y_train, y_train_predict))

    val_errors.append(mean_squared_error(y_val, y_val_predict))



best_epoch = np.argmin(val_errors)

best_val_rmse = np.sqrt(val_errors[best_epoch])



plt.annotate('Best model',

             xy=(best_epoch, best_val_rmse),

             xytext=(best_epoch, best_val_rmse + 1),

             ha="center",

             arrowprops=dict(facecolor='black', shrink=0.05),

             fontsize=16,

            )



best_val_rmse -= 0.03  # just to make the graph look better

plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)

plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")

plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")

plt.legend(loc="upper right", fontsize=14)

plt.xlabel("Epoch", fontsize=14)

plt.ylabel("RMSE", fontsize=14)

plt.title(" Early Stopping")

plt.show()
# which epoch did we find the best model in?

best_epoch, best_model
t = np.linspace(-10, 10, 100)

sig = 1 / (1 + np.exp(-t))

plt.figure(figsize=(9, 3))

plt.plot([-10, 10], [0, 0], "k-")

plt.plot([-10, 10], [0.5, 0.5], "k:")

plt.plot([-10, 10], [1, 1], "k:")

plt.plot([0, 0], [-1.1, 1.1], "k-")

plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")

plt.xlabel("t")

plt.legend(loc="upper left", fontsize=20)

plt.axis([-10, 10, -0.1, 1.1])

plt.title("Logistic Function")

plt.show()
from sklearn import datasets

iris = datasets.load_iris()

list(iris.keys())
print(iris.DESCR)
# Let's use Petal Width as one of the features

X = iris["data"][:,3:]

y = (iris["target"] == 2).astype(np.int) 
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver="lbfgs", random_state=42)

log_reg.fit(X, y)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)

y_proba = log_reg.predict_proba(X_new)



decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]



plt.figure(figsize=(8, 3))

plt.plot(X[y==0], y[y==0], "bs")

plt.plot(X[y==1], y[y==1], "g^")

plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)

plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")

plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")

plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")

plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')

plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')

plt.xlabel("Petal width (cm)", fontsize=14)

plt.ylabel("Probability", fontsize=14)

plt.legend(loc="center left", fontsize=14)

plt.axis([0, 3, -0.02, 1.02])

plt.title("Decision Boundary Analysis")

plt.show()
decision_boundary # So anything with a petal width > 1.66 is an Iris Virginica
log_reg.predict([[1.7], [1.5]]) # Let's test this out
from sklearn.linear_model import LogisticRegression



X = iris["data"][:, (2, 3)]  # petal length, petal width

y = (iris["target"] == 2).astype(np.int)



log_reg = LogisticRegression(solver="lbfgs", C=10**10, random_state=42)

log_reg.fit(X, y)



x0, x1 = np.meshgrid(

        np.linspace(2.9, 7, 500).reshape(-1, 1),

        np.linspace(0.8, 2.7, 200).reshape(-1, 1),

    )

X_new = np.c_[x0.ravel(), x1.ravel()]



y_proba = log_reg.predict_proba(X_new)



plt.figure(figsize=(10, 4))

plt.plot(X[y==0, 0], X[y==0, 1], "bs")

plt.plot(X[y==1, 0], X[y==1, 1], "g^")



zz = y_proba[:, 1].reshape(x0.shape)

contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)





left_right = np.array([2.9, 7])

boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]



plt.clabel(contour, inline=1, fontsize=12)

plt.plot(left_right, boundary, "k--", linewidth=3)

plt.text(3.5, 1.5, "Not Iris virginica", fontsize=14, color="b", ha="center")

plt.text(6.5, 2.3, "Iris virginica", fontsize=14, color="g", ha="center")

plt.xlabel("Petal length", fontsize=14)

plt.ylabel("Petal width", fontsize=14)

plt.axis([2.9, 7, 0.8, 2.7])

plt.title("Linear Decision Boundary")

plt.show()
X = iris["data"][:, (2, 3)]  # petal length, petal width

y = iris["target"] # Trying out all 3 classes here



softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)

softmax_reg.fit(X, y)


x0, x1 = np.meshgrid(

        np.linspace(0, 8, 500).reshape(-1, 1),

        np.linspace(0, 3.5, 200).reshape(-1, 1),

    )

X_new = np.c_[x0.ravel(), x1.ravel()]





y_proba = softmax_reg.predict_proba(X_new)

y_predict = softmax_reg.predict(X_new)



zz1 = y_proba[:, 1].reshape(x0.shape)

zz = y_predict.reshape(x0.shape)



plt.figure(figsize=(10, 4))

plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")

plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")

plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")



from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])



plt.contourf(x0, x1, zz, cmap=custom_cmap)

contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)

plt.clabel(contour, inline=1, fontsize=12)

plt.xlabel("Petal length", fontsize=14)

plt.ylabel("Petal width", fontsize=14)

plt.legend(loc="center left", fontsize=14)

plt.axis([0, 7, 0, 3.5])

plt.title("Softmax Regression Decision Boundary")

plt.show()
softmax_reg.predict([[5, 2]]) # A petal length of 5 and a petal width of 2 should be predicted as Iris Virginica
softmax_reg.predict_proba([[5, 2]]) # How are the class probabilities for this case?
X = iris["data"][:, (2, 3)]  # petal length, petal width

y = iris["target"]
# Add bias

X_with_bias = np.c_[np.ones(([len(X), 1])), X]
# Set Random Seed

np.random.seed(2042)
# Split Train, Val and Test Sets

test_ratio = 0.2

val_ratio = 0.2



total_size = len(X_with_bias)

test_size = int(test_ratio * total_size)

val_size = int(val_ratio * total_size)

train_size = total_size - test_size - val_size



rnd_indices = np.random.permutation(total_size)



train_idx = rnd_indices[:train_size]

test_idx = rnd_indices[-test_size:]

val_idx = rnd_indices[train_size:-test_size]



X_train, y_train = X_with_bias[train_idx], y[train_idx]

X_val, y_val = X_with_bias[val_idx], y[val_idx]

X_test, y_test = X_with_bias[test_idx], y[test_idx]

def to_one_hot(y):

    num_classes = y.max() + 1

    m = len(y)

    y_one_hot = np.zeros((m, num_classes))

    y_one_hot[np.arange(m), y] = 1

    return y_one_hot
# Let's test this out

y_train[:10], to_one_hot(y_train[:10])
y_train_one_hot = to_one_hot(y_train)

y_test_one_hot = to_one_hot(y_test)

y_val_one_hot = to_one_hot(y_val)
def softmax(logits):

    exps = np.exp(logits) # Step 1. Exponentiate

    exp_sums = np.sum(exps, axis=1, keepdims=True)

    return exps / exp_sums
num_inputs = X_train.shape[1] # This is 3 since we have two features, petal width and height and the bias term

num_outputs = len(np.unique(y_train)) # 3 classes of the iris
eta = 0.01 # Learning Rate

num_iterations = 5001 # Number of iterations to train

m = len(X_train) # Number of training instances

eps = 1e-7 # small number to help with stability



theta = np.random.randn(num_inputs, num_outputs) # random initialization of parameters



for iteration in range(num_iterations):

    logits = X_train.dot(theta)

    y_proba = softmax(logits)

    loss = -np.mean(np.sum(y_train_one_hot * np.log(y_proba + eps), axis=1))

    error = y_proba - y_train_one_hot 

    gradients = (1/m) * X_train.T.dot(error)

    if iteration % 500 == 0:

        print("Iteration {} \t Loss: {}".format(iteration, loss))

    theta = theta - eta * gradients
theta # Final Model Parameters
# Predictions on the validation set

y_val_proba = softmax(X_val.dot(theta))

y_predict = np.argmax(y_val_proba, axis=1)



accuracy_score = np.mean(y_val == y_predict)

accuracy_score
eta = 0.01

num_iterations = 5001

m = len(X_train)

eps = 1e-7

alpha = 0.1 # Regularization hyperparameter



theta = np.random.randn(num_inputs, num_outputs)



for iteration in range(num_iterations):

    logits = X_train.dot(theta)

    y_proba = softmax(logits)

    entropy_loss = -np.mean(np.sum(y_train_one_hot * np.log(y_proba + eps), axis=1))

    reg_term = 0.5 * np.sum(np.square(theta[1:])) # Ignore the bias term for regularizing weights

    loss = entropy_loss + alpha * reg_term

    error = y_proba - y_train_one_hot

    gradients = (1/m) * X_train.T.dot(error) + np.r_[np.zeros([1, num_outputs]), alpha * theta[1:]] # Add the regularization here as well

    if iteration % 500 == 0:

        print("Iteration {} \t Loss: {}".format(iteration, loss))

    theta = theta - eta * gradients
# Predictions on the validation set

y_val_proba = softmax(X_val.dot(theta))

y_predict = np.argmax(y_val_proba, axis=1)



accuracy_score = np.mean(y_val == y_predict)

accuracy_score 
eta = 0.1 

num_iterations = 5001

m = len(X_train)

eps = 1e-7

alpha = 0.1  # regularization hyperparameter

best_loss = np.infty



theta = np.random.randn(num_inputs, num_outputs)



for iteration in range(num_iterations):

    logits = X_train.dot(theta)

    y_proba = softmax(logits)

    entropy_loss = -np.mean(np.sum(y_train_one_hot * np.log(y_proba + eps), axis=1))

    reg_term = 0.5 * np.sum(np.square(theta[1:]))

    loss = entropy_loss + alpha * reg_term

    error = y_proba - y_train_one_hot

    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, num_outputs]), alpha * theta[1:]]

    theta = theta - eta * gradients



    logits = X_val.dot(theta)

    y_proba = softmax(logits)

    entropy_loss = -np.mean(np.sum(y_val_one_hot * np.log(y_proba + eps), axis=1))

    reg_term = 0.5 * np.sum(np.square(theta[1:]))

    loss = entropy_loss + alpha * reg_term

    if iteration % 500 == 0:

        print("Iteration {} \t Loss: {}".format(iteration, loss))

    if loss < best_loss:

        best_loss = loss

    else:

        print(" Prev Iteration {} \t Best Loss: {}".format(iteration - 1, best_loss))

        print("Iteration {} \t Loss: {} Early Stopping!".format(iteration, loss))

        break
logits = X_val.dot(theta)

y_proba = softmax(logits)

y_predict = np.argmax(y_proba, axis=1)



accuracy_score = np.mean(y_predict == y_val)

accuracy_score
x0, x1 = np.meshgrid(

        np.linspace(0, 8, 500).reshape(-1, 1),

        np.linspace(0, 3.5, 200).reshape(-1, 1),

    )

X_new = np.c_[x0.ravel(), x1.ravel()]

X_new_with_bias = np.c_[np.ones([len(X_new), 1]), X_new]



logits = X_new_with_bias.dot(theta)

y_proba = softmax(logits)

y_predict = np.argmax(y_proba, axis=1)



zz1 = y_proba[:, 1].reshape(x0.shape)

zz = y_predict.reshape(x0.shape)



plt.figure(figsize=(10, 4))

plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")

plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")

plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")



from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])



plt.contourf(x0, x1, zz, cmap=custom_cmap)

contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)

plt.clabel(contour, inline=1, fontsize=12)

plt.xlabel("Petal length", fontsize=14)

plt.ylabel("Petal width", fontsize=14)

plt.legend(loc="upper left", fontsize=14)

plt.axis([0, 7, 0, 3.5])

plt.show()
logits = X_test.dot(theta)

y_proba = softmax(logits)

y_predict = np.argmax(y_proba, axis=1)



accuracy_score = np.mean(y_predict == y_test)

accuracy_score # Not bad!