import numpy as np
rng = np.random.RandomState(0)
import matplotlib.pyplot as plt


def wave(n_points=100, n_points_perturbed=20, pertubation_mult=1):
    X = 5 * rng.rand(n_points, 1)
    y = np.sin(X).ravel()
    perturbation_step_size = n_points // n_points_perturbed
    y[::perturbation_step_size] += 3 * pertubation_mult * (0.5 - rng.rand(X.shape[0] // perturbation_step_size))
    
    # Sort in order of X.
    sort = np.argsort(X[:, 0])
    X = X[sort]
    y = y[sort]
    
    return X, y


X, y = wave(n_points=100, n_points_perturbed=20, pertubation_mult=1)

from sklearn.gaussian_process import GaussianProcessRegressor
clf = GaussianProcessRegressor(random_state=42)
clf.fit(X, y)
y_pred = clf.predict(X)

fig, ax = plt.subplots(1, figsize=(12, 6))
plt.plot(X[:, 0], y, marker='o', color='black', linewidth=0)
plt.plot(X[:, 0], y_pred, marker='x', color='steelblue')
plt.suptitle("$GaussianProcessRegressor(kernel=RBF)$ [default]", fontsize=20)
plt.axis('off')
pass