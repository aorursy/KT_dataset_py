from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
BASEPATH = "../input/osic-pulmonary-fibrosis-ct-scans"

ct_scans = os.listdir(BASEPATH)
def display_ct_scan(path):

    image = np.load(os.path.join(BASEPATH, path))

    dims = image.shape

    idxs = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    fig = plt.figure(figsize=(18,5))

    for i, idx in enumerate(idxs):

        ax = plt.subplot(1,6,i+1)

        slice = int(idx * dims[0]) - 1

        plt.imshow(image[slice,:,:], cmap='bone')

        plt.axis('off')

        plt.suptitle(f'CT scan of {path.split(".")[0]}')
display_ct_scan(ct_scans[0])

display_ct_scan(ct_scans[100])