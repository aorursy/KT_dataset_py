import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.filters import threshold_adaptive

sample_files = ['../input/train/101.png', '../input/train/11.png', '../input/train/120.png', '../input/train/155.png', '../input/train/164.png']

def denoiseimage(inp_path):
    img = rgb2gray(imread(inp_path))
    block_size = 35
    #apply adaptive threshold.
    binary_adaptive = threshold_adaptive(img, block_size, offset=10)
    return binary_adaptive

fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(25,40))
for index, file in enumerate(sample_files):
    noise_reduced_file = denoiseimage(file)
    ax[index][0].imshow(imread(file), cmap="gray")
    ax[index][1].imshow(noise_reduced_file, cmap="gray")
    
plt.tight_layout()
plt.show()
from skimage.morphology import binary_closing

fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(25,40))
kernel=[[1,1],[1,1]]
for index, file in enumerate(sample_files):
    noise_reduced_file = binary_closing(denoiseimage(file), kernel)
    ax[index][0].imshow(imread(file), cmap="gray")
    ax[index][1].imshow(noise_reduced_file, cmap="gray")
    
plt.tight_layout()
plt.show()
smimg = denoiseimage(sample_files[4])

def horizontal_projections(sobel_image):
    sum_of_cols = []
    rows,cols = sobel_image.shape
    for row in range(rows-1):
        sum_of_cols.append(np.sum(sobel_image[row,:]))
    return sum_of_cols

hp = horizontal_projections(smimg)
plt.plot(hp)
for index, sump in enumerate(hp):
    if sump > 430:
        smimg[index,:] = 1