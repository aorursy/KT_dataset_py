import os

import karta

import numpy as np

from skimage.exposure import equalize_adapthist

import matplotlib.pyplot as plt

%matplotlib inline
grids = []

errors = []



for fnm in filter(lambda a: a.endswith(".tif"), os.listdir("train-tif-sample/")):

    try:

        grid = karta.read_gtiff("train-tif-sample/{}".format(fnm))

        grids.append((fnm, grid))

    except IOError as e:

        errors.append((fnm, e))
fig = plt.figure(figsize=(12, 12))



i = 0

while i != 16:

    fnm, grid = grids[i]

    ax = fig.add_subplot(4, 4, i+1)

    a = np.dstack([equalize_adapthist(grid[::-1,:,i]) for i in (2, 1, 0)])

    ax.imshow(a)

    ax.set_title(fnm)

    ax.set_xticks([])

    ax.set_yticks([])

    i += 1
errors