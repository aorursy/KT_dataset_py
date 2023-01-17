import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



import pandas as pd

df = pd.read_csv("../input/ted-talks/ted_main.csv")

df.head()
df['duration'].plot.hist(20)
import seaborn as sns

sns.kdeplot(df['duration'])
import numpy as np

X = df['duration'].values[:, np.newaxis]

from sklearn.neighbors.kde import KernelDensity

kde10 = KernelDensity(kernel='gaussian', bandwidth=10)

kde10.fit(X)

kde40 = KernelDensity(kernel='gaussian', bandwidth=40)

kde40.fit(X)

kde100 = KernelDensity(kernel='gaussian', bandwidth=100)

kde100.fit(X)
n2000 = np.array(list(range(0, 2001)))[:, np.newaxis]



fig, ax = plt.subplots(1, figsize=(12, 6))

ax.plot(n2000, np.exp(kde10.score_samples(n2000)), label='10-band')

ax.plot(n2000, np.exp(kde40.score_samples(n2000)), label='40-band')

ax.plot(n2000, np.exp(kde100.score_samples(n2000)), label='100-band')

ax.legend()

ax.set_title("TED Talk Speech Lengths, KDE Estimates by Bandwidth (Gaussian)")
kde10tophat = KernelDensity(kernel='tophat', bandwidth=10).fit(X)

kde40tophat = KernelDensity(kernel='tophat', bandwidth=40).fit(X)



fig, ax = plt.subplots(1, figsize=(12, 6))

ax.plot(n2000, np.exp(kde10tophat.score_samples(n2000)), label='10-band')

ax.plot(n2000, np.exp(kde40tophat.score_samples(n2000)), label='40-band')

ax.legend()

ax.set_title("TED Talk Speech Lengths, KDE Estimates by Bandwidth (Tophat)")
from sklearn.datasets import load_digits

from sklearn.neighbors import KernelDensity

from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV



# load the data

digits = load_digits()

data = digits.data



# project the 64-dimensional data to a lower dimension

pca = PCA(n_components=15, whiten=False)

data = pca.fit_transform(digits.data)



# use grid search cross-validation to optimize the bandwidth

params = {'bandwidth': np.logspace(-1, 1, 20)}

grid = GridSearchCV(KernelDensity(), params)

grid.fit(data)



print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))



# use the best estimator to compute the kernel density estimate

kde = grid.best_estimator_



# sample 44 new points from the data

new_data = kde.sample(44, random_state=0)

new_data = pca.inverse_transform(new_data)



# turn data into a 4x11 grid

new_data = new_data.reshape((4, 11, -1))

real_data = digits.data[:44].reshape((4, 11, -1))



# plot real digits and resampled digits

fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))

for j in range(11):

    ax[4, j].set_visible(False)

    for i in range(4):

        im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)),

                             cmap=plt.cm.binary, interpolation='nearest')

        im.set_clim(0, 16)

        im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)),

                                 cmap=plt.cm.binary, interpolation='nearest')

        im.set_clim(0, 16)



ax[0, 5].set_title('Selection from the input data')

ax[5, 5].set_title('"New" digits drawn from the kernel density model')



plt.show()