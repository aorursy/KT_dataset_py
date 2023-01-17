import numpy as np
import pandas as pd
import mlcrate as mlc
from skimage.io import imread, imsave
from skimage.transform import resize
# Download and display our image
from urllib import request
from io import BytesIO
import matplotlib.pyplot as plt
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/250px-Gatto_europeo4.jpg'
data = BytesIO(request.urlopen(url).read())
image = imread(data, format='jpg')

plt.imshow(image)
img = image[:, :, 0]
plt.imshow(img, cmap='gray')
# Load Kuzushiji-49 and bin characters by mean grey level
imgs_train = np.load('../input/k49-train-imgs.npz')['arr_0']
means = (255 - imgs_train).mean(axis=1).mean(axis=1)
plt.hist(means, bins=50, log=True)

from collections import defaultdict
character_bins = defaultdict(list)
for char, mean in zip(imgs_train, means):
    character_bins[int(mean)].append(char)
OUTPUT_RESOLUTION = (96, 75) # Note this is multiplied by 28x28 since each pixel is replaced by a character
img_small = (resize(img, OUTPUT_RESOLUTION, preserve_range=True) / 2) + 120 # We rescale pixels to (120, 245) since K49 doesn't have characters with lower gray levels

plt.hist(img_small.flatten(), bins=50, log=True)
plt.show()
plt.imshow(img_small, cmap='gray')
None
new_img = np.zeros((img_small.shape[0]*28, img_small.shape[1]*28), dtype=np.uint8)
# Loop over all pixels in original image, selecting a random character from K49 with same gray level, and putting that character in the same position in the new image
ix = 0
iy = 0
for iy in range(img_small.shape[0]):
    for ix in range(img_small.shape[1]):
        level = int(img_small[iy, ix])
        charbin = character_bins[level]
        if len(charbin) == 0:
            charbin = character_bins[level + 1]
        if len(charbin) == 0:
            charbin = character_bins[level - 1]
        char = 255 - charbin[np.random.choice(np.arange(len(charbin)))]
        new_img[iy*28:(iy+1)*28, ix*28:(ix+1)*28] = char
plt.figure(figsize=(20, 20))
plt.imshow(new_img, cmap='gray')
imsave('cat.png', new_img)