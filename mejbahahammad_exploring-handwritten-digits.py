from sklearn.manifold import Isomap

from sklearn.datasets import load_digits

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

%matplotlib inline
digits_data = load_digits()

digits_data.images.shape
fig, axes = plt.subplots(10, 10, figsize= (10, 8),

                      subplot_kw = {'xticks':[], 'yticks':[]},

                      gridspec_kw = dict(hspace = 0.1, wspace = 0.1))



for i, ax in enumerate(axes.flat):

    ax.imshow(digits_data.images[i], cmap = plt.cm.binary, interpolation = 'nearest')

    ax.text(0.05, 0.05, str(digits_data.target[i]),

           transform = ax.transAxes, color = 'green')
iso = Isomap(n_components = 2)

iso.fit(digits_data.data)

data_projected = iso.transform(digits_data.data)

data_projected.shape
plt.scatter(data_projected[:, 0], data_projected[:, 1], c = digits_data.target,

           edgecolors='none', alpha=0.5,

           cmap = plt.get_cmap('nipy_spectral', 10))

plt.colorbar(label = "digit label", ticks = range(10))

plt.clim(-0.5, 9.5)

plt.tight_layout()

plt.show()
X = digits_data.data

y = digits_data.target
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 0)
model = GaussianNB()
model.fit(x_train, y_train)
y_model = model.predict(x_test)
accuracy_score(y_test, y_model)
matrix = confusion_matrix(y_test, y_model)



plt.figure(figsize = (10, 8))

sns.heatmap(matrix, square = True, annot = True, cbar = False)

plt.xlabel("Predicted Value")

plt.ylabel("True Value")
fig, axes = plt.subplots(10, 10, figsize= (10, 8),

                      subplot_kw = {'xticks':[], 'yticks':[]},

                      gridspec_kw = dict(hspace = 0.1, wspace = 0.1))



for i, ax in enumerate(axes.flat):

    ax.imshow(digits_data.images[i], cmap = plt.cm.binary, interpolation = 'nearest')

    ax.text(0.05, 0.05, str(digits_data.target[i]),

           transform = ax.transAxes, color = 'blue' if (y_test[i] == y_model[i]) else 'red')