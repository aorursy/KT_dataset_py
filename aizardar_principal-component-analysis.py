import numpy as np

import matplotlib.pyplot as plt





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



plt.figure(figsize=(8,8))

#plt.subplot2grid((3,2), (0, 0), rowspan=3)

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
# 1. Solid line



plt.plot([-2, 2], [0, 0], "k-", linewidth=1)

plt.plot(X_proj1[:, 0], np.zeros(m), "bo", alpha=0.3)

plt.gca().get_yaxis().set_ticks([])

plt.axis([-2, 2, -1, 1])

plt.xlabel("$z_1$", fontsize=18)

plt.grid(True)
# 2. Dashed line



plt.plot([-2, 2], [0, 0], "k--", linewidth=1)

plt.plot(X_proj2[:, 0], np.zeros(m), "bo", alpha=0.3)

plt.gca().get_yaxis().set_ticks([])

plt.axis([-2, 2, -1, 1])

plt.xlabel("$z_1$", fontsize=18)

plt.grid(True)
# 3. Dotted line



plt.plot([-2, 2], [0, 0], "k:", linewidth=1)

plt.plot(X_proj3[:, 0], np.zeros(m), "bo", alpha=0.3)

plt.gca().get_yaxis().set_ticks([])

plt.axis([-2, 2, -1, 1])

plt.xlabel("$z_1$", fontsize=18)

plt.grid(True)
from sklearn.datasets import load_digits

from sklearn.decomposition import PCA



digits = load_digits()

X = digits.data

y = digits.target



pca = PCA(2)  # project from 64 to 2 dimensions

X_proj = pca.fit_transform(X)

print(X.shape)

print(X_proj.shape)



# Let's plot 



plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, edgecolor='none', alpha=0.5,

            cmap=plt.cm.get_cmap('nipy_spectral', 10))

plt.colorbar();
fig, axes = plt.subplots(8, 8, figsize=(8, 8))

fig.subplots_adjust(hspace=0.1, wspace=0.1)



for i, ax in enumerate(axes.flat):

    pca = PCA(i + 1).fit(X)

    im = pca.inverse_transform(pca.transform(X[15:16]))   # Adjust here to see another digit



    ax.imshow(im.reshape((8, 8)), cmap='binary')

    ax.text(0.95, 0.05, 'n = {0}'.format(i + 1), ha='right',

            transform=ax.transAxes, color='green')

    ax.set_xticks([])

    ax.set_yticks([])