from zipfile import ZipFile

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as img

from mpl_toolkits.axes_grid1 import ImageGrid

from PIL import Image

import cv2
data = np.load("/kaggle/input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/East_China.npy")
fig = plt.figure(figsize=(15., 15.))

grid = ImageGrid(fig, 111,  # similar to subplot(111)

                 nrows_ncols=(3, 5),  # creates 2x2 grid of axes

                 axes_pad=0.1,  # pad between axes in inch.

                 )



for ax, im in zip(grid, range(15)):

    # Iterating over the grid returns the Axes.

    ax.imshow(data[im].astype(np.uint8))
data.shape
whiteintensitythreshold = 230

numwhitethreshold = 70

interestingregions = pd.DataFrame([(i, j, numwhite) for i, j, numwhite in 

           ((i,j, np.sum(data[0, i:(i+30), j:(j+31)]>whiteintensitythreshold)) 

              for i in range(data.shape[1]-30) for j in range(data.shape[2]-31)) if numwhite>numwhitethreshold], 

                                  columns=["i","j", "numwhite"])
from sklearn.cluster import OPTICS

clustering = OPTICS(min_samples=50).fit(interestingregions[["i", "j"]])

print(set(clustering.labels_))

interestingregions["clusid"] = clustering.labels_
interestingregions.loc[interestingregions.groupby(["clusid"])["numwhite"].idxmax()]
plt.imshow(data[0].astype(np.uint8))


for idx, row in interestingregions.loc[interestingregions.groupby(["clusid"])["numwhite"].idxmax()].iterrows():

    cX = int(row["i"]+20)

    cY = int(row["j"]+20)

    radius = 20

    cv2.circle(data[0], (int(cY), int(cX)), int(radius), (255, 255,  0), 3)

    

plt.imshow(data[0].astype(np.uint8))