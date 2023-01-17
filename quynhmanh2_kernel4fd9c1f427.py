import h5py

import numpy as np

D = h5py.File('../input/breast.h5', 'r')

X,Y,P = D['images'],np.array(D['counts']),np.array(D['id'])
print("There are", len(X), "training and test examples")
from matplotlib import pyplot as plt

plt.imshow(X[0])

plt.show()

plt.imshow(X[1])

plt.show()

plt.imshow(X[1])

plt.show()

plt.imshow(X[2])

plt.show()

plt.imshow(X[3])

plt.show()

plt.imshow(X[4])

plt.show()
import numpy as np

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt



num_bins = 200

n, bins, patches = plt.hist(Y, num_bins, facecolor='blue', alpha=0.5)

plt.show()



print(Y[:4])
from skimage import data

from skimage.color import rgb2hed

from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt



# Create an artificial color close to the original one

cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])

cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white',

                                             'saddlebrown'])

cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['darkviolet',

                                               'white'])



ihc_rgb = X[0]

ihc_hed = rgb2hed(ihc_rgb)



fig, axes = plt.subplots(1, 2, figsize=(7, 6), sharex=True, sharey=True)

ax = axes.ravel()





ax[0].imshow(ihc_rgb)

ax[0].set_title("Original image")



ax[1].imshow(ihc_hed[:, :, 2], cmap=cmap_dab)

ax[1].set_title("DAB " + str(Y[0]))



for a in ax.ravel():

    a.axis('off')

    

ihc_rgb = X[1]

ihc_hed = rgb2hed(ihc_rgb)



fig, axes = plt.subplots(1, 2, figsize=(7, 6), sharex=True, sharey=True)

ax = axes.ravel()



ax[0].imshow(ihc_rgb)

ax[0].set_title("Original image")



ax[1].imshow(ihc_hed[:, :, 2], cmap=cmap_dab)

ax[1].set_title("DAB " + str(Y[1]))



for a in ax.ravel():

    a.axis('off')

    



ihc_rgb = X[2]

ihc_hed = rgb2hed(ihc_rgb)



fig, axes = plt.subplots(1, 2, figsize=(7, 6), sharex=True, sharey=True)

ax = axes.ravel()



ax[0].imshow(ihc_rgb)

ax[0].set_title("Original image")



ax[1].imshow(ihc_hed[:, :, 2], cmap=cmap_dab)

ax[1].set_title("DAB " + str(Y[2]))



for a in ax.ravel():

    a.axis('off')

    

    ihc_rgb = X[3]

ihc_hed = rgb2hed(ihc_rgb)



fig, axes = plt.subplots(1, 2, figsize=(7, 6), sharex=True, sharey=True)

ax = axes.ravel()



ax[0].imshow(ihc_rgb)

ax[0].set_title("Original image")



ax[1].imshow(ihc_hed[:, :, 2], cmap=cmap_dab)

ax[1].set_title("DAB " + str(Y[3]))



for a in ax.ravel():

    a.axis('off')

    

ihc_rgb = X[4]

ihc_hed = rgb2hed(ihc_rgb)



fig, axes = plt.subplots(1, 2, figsize=(7, 6), sharex=True, sharey=True)

ax = axes.ravel()



ax[0].imshow(ihc_rgb)

ax[0].set_title("Original image")



ax[1].imshow(ihc_hed[:, :, 2], cmap=cmap_dab)

ax[1].set_title("DAB " + str(Y[4]))



for a in ax.ravel():

    a.axis('off')

    

    ihc_rgb = X[5]

ihc_hed = rgb2hed(ihc_rgb)



fig, axes = plt.subplots(1, 2, figsize=(7, 6), sharex=True, sharey=True)

ax = axes.ravel()



ax[0].imshow(ihc_rgb)

ax[0].set_title("Original image")



ax[1].imshow(ihc_hed[:, :, 2], cmap=cmap_dab)

ax[1].set_title("DAB " + str(Y[5]))



for a in ax.ravel():

    a.axis('off')



import numpy as np

np.mean(X[0], axis=(0, 1))



ihc_rgb = X[0]

ihc_hed = rgb2hed(ihc_rgb)



np.mean(ihc_hed[:, :, 2])



result = []

result2 = []

result3 = []

result4 = []

n = len(Y)

for i in range(0,13):

    res = np.mean(X[i], axis=(0, 1))

    res2 = np.mean(ihc_hed[:, :, 2])

    result.append(np.append(res, [res2]))

    

    res3 = np.var(X[i], axis=(0, 1))

    res4 = np.var(ihc_hed[:, :, 2])

    result2.append(np.append(res3, [res4]))

print("average of the “brown”, red, green and blue channels")

print(result)

print("variance of the “brown”, red, green and blue channels")

print(result2)

import statsmodels.api as sm

import matplotlib.pyplot as plt

from statsmodels.sandbox.regression.predstd import wls_prediction_std



np.random.seed(9876789)



nsample = 100

x = np.linspace(0, 10, 100)

X = np.column_stack((x, x**2))

beta = np.array([1, 0.1, 10])

e = np.random.normal(size=nsample)



X = sm.add_constant(X)

y = np.dot(X, beta) + e



model = sm.OLS(y, X)

results = model.fit()

print(results.summary())

print('Parameters: ', results.params)

print('R2: ', results.rsquared)