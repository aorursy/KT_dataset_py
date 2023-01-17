from sklearn.decomposition import PCA
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data
y = iris.target

n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
import matplotlib.pyplot as plt


colors = ['yellow', 'lime', 'red']

plt.figure(figsize=(8, 8))
for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                color=color, lw=2, label=target_name)

plt.title('PCA - Iris dataset')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.axis([-4, 4, -1.5, 1.5])

plt.show()