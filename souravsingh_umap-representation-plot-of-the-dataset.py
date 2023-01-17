import numpy as np
import pandas as pd

import os
from skimage.io import imread
from glob import glob

import itertools
import shutil
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
%matplotlib inline
base_tile_dir = '../input/train/'
df = pd.DataFrame({'path': glob(os.path.join(base_tile_dir,'*.tif'))})
df['id'] = df.path.map(lambda x: x.split('/')[3].split(".")[0])
labels = pd.read_csv("../input/train_labels.csv")
df = df.merge(labels, on = "id")
df.head(5)
df0 = df[df.label == 0].sample(10000, random_state = 42)
df1 = df[df.label == 1].sample(10000, random_state = 42)
df = pd.concat([df0, df1], ignore_index=True).reset_index()
df = df[["path", "id", "label"]]
df.sample(10)
df['image'] = df['path'].map(imread)
df.sample(3)
input_images = np.stack(list(df.image), axis = 0)
input_images.shape
encoder = LabelBinarizer()
y = encoder.fit_transform(df.label)
nsamples, nx, ny, nz = input_images.shape
x = input_images.reshape((nsamples,nx*ny*nz))
x.shape
import umap
embedding = umap.UMAP(n_components=2, min_dist=0.3, metric='correlation',random_state=42, verbose=True).fit_transform(x)
fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(
    embedding[:, 0], embedding[:, 1], cmap="Spectral", s=0.1
)
plt.title("Cancer detection data embedded into two dimensions by UMAP", fontsize=18)

plt.show()
