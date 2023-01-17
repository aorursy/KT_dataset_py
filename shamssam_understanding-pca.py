import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
X = np.zeros((50, 2))
X[:, 0] = np.linspace(0, 50)
X[:, 1] = X[:, 0]
X = X + 5*np.random.randn(X.shape[0], X.shape[1])
plt.scatter(X[:, 0], X[:, 1], c='g', alpha='0.5')
plt.xlabel('x1')
plt.xlabel('x2')
plt.axis('scaled')
plt.show()
pca = PCA(2)
pca_reduce = PCA(1)
X_proj = pca_reduce.fit_transform(X)
X_rebuild = pca_reduce.inverse_transform(X_proj)
X_proj = pca.fit_transform(X)
plt.figure(figsize=(7, 7))

# plot data and projection
plt.scatter(X[:,0], X[:, 1], alpha=0.5, c='green')
plt.scatter(X_rebuild[:, 0], X_rebuild[:, 1], alpha=0.3, c='r')

# plot the components
# scaled to the length of their variance i.e. eigenvalues
soa = np.hstack((
    np.ones(pca.components_.shape) * pca.mean_, 
    pca.components_ * np.atleast_2d(
        np.sqrt(pca.explained_variance_)
    ).transpose()
))
x, y, u, v = zip(*soa)

ax = plt.gca()
ax.quiver(
    x, y, u, v, 
    angles='xy', 
    scale_units='xy', 
    scale=0.5, 
    color='rb'
)
plt.axis('scaled')
plt.draw()

plt.legend([
    'original', 
    'projection'
])

# plot the projection errors
for p_orig, p_proj in zip(X, X_rebuild):
    plt.plot([p_orig[0], p_proj[0]], [p_orig[1], p_proj[1]], c='g', alpha=0.3)
    
plt.show()
lin_reg = LinearRegression()
lin_reg.fit(X[:,0].reshape(-1, 1), X[:, 1])
y_pred = X[:,0] * lin_reg.coef_
plt.figure(figsize=(7, 7))

# plot data and projection
plt.scatter(X[:,0], X[:, 1], alpha=0.5, c='green')
plt.scatter(X_rebuild[:, 0], X_rebuild[:, 1], alpha=0.3, c='r')
plt.scatter(X[:,0], y_pred, alpha=0.5, c='blue')

# plot the projection errors
for p_orig, p_proj in zip(X, X_rebuild):
    plt.plot([p_orig[0], p_proj[0]], [p_orig[1], p_proj[1]], c='r', alpha=0.3)

# plot the prediction errors
for p_orig, y in zip(X, np.hstack((X[:,0].reshape(-1, 1), y_pred.reshape(-1, 1)))):
    plt.plot([p_orig[0], y[0]], [p_orig[1], y[1]], c='b', alpha=0.3)