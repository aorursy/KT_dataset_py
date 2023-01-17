import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
sns.set_style('darkgrid')

sns.set_context('talk')

%config InlineBackend.figure_format = 'retina'
x = np.linspace(0, 10, 100)

y = np.sin(0.5 * x) + np.sin(2 * x)

x_test = np.linspace(10, 12, 20)

y_test = np.sin(0.5 * x_test) + np.sin(2 * x_test)
plt.scatter(x, y, label='train')

plt.scatter(x_test, y_test, label='test')

plt.legend();
idxs = np.random.permutation(np.arange(100))

valid_idxs = idxs[:20]

train_idxs = idxs[20:]

x_train = x[train_idxs]

y_train = y[train_idxs]

x_valid = x[valid_idxs]

y_valid = y[valid_idxs]
plt.scatter(x_train, y_train, label='train')

plt.scatter(x_test, y_test, label='test')

plt.scatter(x_valid, y_valid, label='valid')

plt.legend();
idxs = np.arange(100)

valid_idxs = idxs[-20:]

train_idxs = idxs[:-20]

x_train = x[train_idxs]

y_train = y[train_idxs]

x_valid = x[valid_idxs]

y_valid = y[valid_idxs]
plt.scatter(x_train, y_train, label='train')

plt.scatter(x_test, y_test, label='test')

plt.scatter(x_valid, y_valid, label='valid')

plt.legend();
x = np.random.uniform(size=30)

y = x + np.random.randn(30) * 0.2



x_valid = np.random.uniform(size=10)

y_valid = x_valid + np.random.randn(10) * 0.2
plt.scatter(x, y)

plt.scatter(x_valid, y_valid);
dt = DecisionTreeRegressor(min_samples_leaf=1)

dt.fit(x[:, None], y)
xx = np.linspace(0, 1, 500)

preds = dt.predict(xx[:, None])
plt.scatter(x, y)

plt.scatter(x_valid, y_valid)

plt.plot(xx, preds, c='g');