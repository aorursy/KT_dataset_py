import os

import glob

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
%%time

# Read all dataset

dataset = {}

for target in sorted(glob.glob("/kaggle/input/data/*")):

    target_name = target.split('/')[-1]

    filenames = sorted(glob.glob(target + '/*'))

    dataset[target_name] = np.array([np.array(Image.open(f)) for f in filenames]).transpose(1, 2, 0)

    

print(dataset.keys())
# 1. coffee

print("Size of HSI: (height, width, spectral) = ", dataset["coffee"].shape)

mean_img = np.mean(dataset["coffee"], axis=2)

plt.imshow(mean_img)
# 2. rice

print("Size of HSI: (height, width, spectral) = ", dataset["rice"].shape)

mean_img = np.mean(dataset["rice"], axis=2)

plt.imshow(mean_img)
# 3. sugar_salt_flour

print("Size of HSI: (height, width, spectral) = ", dataset["sugar_salt_flour"].shape)

mean_img = np.mean(dataset["sugar_salt_flour"], axis=2)

plt.imshow(mean_img)
# 4. sugar_salt_flour_contamination

print("Size of HSI: (height, width, spectral) = ", dataset["sugar_salt_flour_contamination"].shape)

mean_img = np.mean(dataset["sugar_salt_flour_contamination"], axis=2)

plt.imshow(mean_img)
# 5. yatsuhashi

print("Size of HSI: (height, width, spectral) = ", dataset["yatsuhashi"].shape)

mean_img = np.mean(dataset["yatsuhashi"], axis=2)

plt.imshow(mean_img)