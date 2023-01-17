import pandas as pd
df = pd.read_csv("../input/WorldCupMatches.csv")
df = (df
          .loc[df.apply(lambda srs: ((srs['Home Team Name'] == "Brazil") & (srs['Home Team Goals'] >= srs['Away Team Goals'])) | 
                        ((srs['Away Team Name'] == "Brazil") & (srs['Away Team Goals'] >= srs['Home Team Goals'])), axis='columns')]
          .Datetime
          .dropna()
          .map(lambda dt: dt.split("-")[0].strip())
          .map(lambda dt: dt.split(" ")[-1])
          .value_counts()
          .sort_index()
)

import numpy as np
X = np.asarray(list(range(len(df))))[:, np.newaxis]
y = df.values

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
df.plot.line(marker="o", linewidth=0, color='black', title='Brazil Wins or Ties Per World Cup, 1930-2016', figsize=(12, 6))
import numpy as np
rng = np.random.RandomState(0)

X_p = 5 * rng.rand(100, 1)
y_p = np.sin(X_p).ravel()

y_p[::5] += 3 * (0.5 - rng.rand(X_p.shape[0] // 5))

fig, ax = plt.subplots(1, figsize=(12, 6))
plt.plot(X_p[:, 0], y_p, marker='o', color='black', linewidth=0)
plt.suptitle("Synthetic Polynomial Data")
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))


clf = KernelRidge(kernel='rbf', gamma=0.1)
clf.fit(X, y)
y_pred = clf.predict(X)
pd.Series(y_pred).plot(ax=axarr[0])
pd.Series(y).plot(ax=axarr[0], marker="o", linewidth=0, color='black')
axarr[0].set_title("Brazil")

clf = Ridge()
clf.fit(X, y)
y_pred = clf.predict(X)
pd.Series(y_pred).plot(ax=axarr[0], color='gray')


axarr[1].plot(X_p[:, 0], y_p, marker='o', color='black', linewidth=0)

clf = KernelRidge(kernel='rbf', gamma=0.8)
clf.fit(X_p, y_p)
y_pred = clf.predict(X_p)
sort = np.argsort(X_p[:, 0])
axarr[1].plot(X_p[:, 0][sort], y_pred[sort], markeredgewidth=0)

clf = Ridge()
clf.fit(X_p, y_p)
y_pred = clf.predict(X_p)
axarr[1].plot(X_p[:, 0], y_pred, color='gray')
axarr[1].set_title("Sine Wave")
pass
from sklearn.tree import DecisionTreeRegressor

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))


clf = DecisionTreeRegressor(max_depth=5)
clf.fit(X, y)
y_pred = clf.predict(X)
pd.Series(y_pred).plot(ax=axarr[0], color='lightsteelblue')
pd.Series(y).plot(ax=axarr[0], marker="o", linewidth=0, color='black')

clf = DecisionTreeRegressor(max_depth=2)
clf.fit(X, y)
y_pred = clf.predict(X)
pd.Series(y_pred).plot(ax=axarr[0], color='steelblue')
pd.Series(y).plot(ax=axarr[0], marker="o", linewidth=0, color='black')

axarr[0].set_title("Brazil")

axarr[1].plot(X_p[:, 0], y_p, marker='o', color='black', linewidth=0)

clf = DecisionTreeRegressor(max_depth=5)
clf.fit(X_p, y_p)
y_pred = clf.predict(X_p)
sort = np.argsort(X_p[:, 0])
axarr[1].plot(X_p[:, 0][sort], y_pred[sort], markeredgewidth=0, color='lightsteelblue')

clf = DecisionTreeRegressor(max_depth=2)
clf.fit(X_p, y_p)
y_pred = clf.predict(X_p)
sort = np.argsort(X_p[:, 0])
axarr[1].plot(X_p[:, 0][sort], y_pred[sort], markeredgewidth=0, color='steelblue')
axarr[1].set_title("Sine Wave")
pass
from sklearn.neighbors import KNeighborsRegressor

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))


clf = KNeighborsRegressor(n_neighbors=5, weights='uniform')
clf.fit(X, y)
y_pred = clf.predict(X)
pd.Series(y_pred).plot(ax=axarr[0], color='steelblue')
pd.Series(y).plot(ax=axarr[0], marker="o", linewidth=0, color='black')

# clf = KNeighborsRegressor(n_neighbors=5, weights='distance')
# clf.fit(X, y)
# y_pred = clf.predict(X)
# pd.Series(y_pred).plot(ax=axarr[0], color='red')
# pd.Series(y).plot(ax=axarr[0], marker="o", linewidth=0, color='black')

axarr[0].set_title("Brazil")

axarr[1].plot(X_p[:, 0], y_p, marker='o', color='black', linewidth=0)

clf = KNeighborsRegressor(n_neighbors=10, weights='uniform')
clf.fit(X_p, y_p)
y_pred = clf.predict(X_p)
sort = np.argsort(X_p[:, 0])
axarr[1].plot(X_p[:, 0][sort], y_pred[sort], markeredgewidth=0, color='steelblue')

# clf = KNeighborsRegressor(n_neighbors=5, weights='distance')
# clf.fit(X_p, y_p)
# y_pred = clf.predict(X_p)
# sort = np.argsort(X_p[:, 0])
# axarr[1].plot(X_p[:, 0][sort], y_pred[sort], markeredgewidth=0, color='steelblue')
axarr[1].set_title("Sine Wave")
pass
from sklearn.svm import SVR

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))


clf = SVR()
clf.fit(X, y)
y_pred = clf.predict(X)
pd.Series(y_pred).plot(ax=axarr[0], color='steelblue')
pd.Series(y).plot(ax=axarr[0], marker="o", linewidth=0, color='black')

axarr[0].set_title("Brazil")

axarr[1].plot(X_p[:, 0], y_p, marker='o', color='black', linewidth=0)

clf = SVR()
clf.fit(X_p, y_p)
y_pred = clf.predict(X_p)
sort = np.argsort(X_p[:, 0])
axarr[1].plot(X_p[:, 0][sort], y_pred[sort], markeredgewidth=0, color='steelblue')

axarr[1].set_title("Sine Wave")
pass
from sklearn.linear_model import SGDRegressor

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))


clf = SGDRegressor()
clf.fit(X, y)
y_pred = clf.predict(X)
pd.Series(y_pred).plot(ax=axarr[0], color='steelblue')
pd.Series(y).plot(ax=axarr[0], marker="o", linewidth=0, color='black')

axarr[0].set_title("Brazil")

axarr[1].plot(X_p[:, 0], y_p, marker='o', color='black', linewidth=0)

clf = SGDRegressor()
clf.fit(X_p, y_p)
y_pred = clf.predict(X_p)
sort = np.argsort(X_p[:, 0])
axarr[1].plot(X_p[:, 0][sort], y_pred[sort], markeredgewidth=0, color='steelblue')

axarr[1].set_title("Sine Wave")
pass