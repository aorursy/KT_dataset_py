import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from pathlib import Path
image_path = Path('../input/train/images')
mask_path = Path('../input/train/masks')
width = 10
height = 300
fig, ax = plt.subplots(height, width, figsize=(20, 500))
for k, path in enumerate(mask_path.glob('*.png')):
    if k >= width * height:
        break
    mask = img_to_array(load_img(path, grayscale=True))
    img = load_img(image_path / path.name)
    ax[k // width, k % width].axis('off')
    ax[k // width, k % width].imshow(img)
    mask = mask.squeeze() / 255
    if np.max(mask) > 0:
        ax[k // width, k % width].contour(mask, colors='r', levels=[0.5], alpha=0.3)
plt.show()