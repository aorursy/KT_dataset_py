from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

import os

%matplotlib inline
BASE_PATH = '../input/gan-getting-started/'

MONET_PATH = os.path.join(BASE_PATH, 'monet_jpg')

PHOTO_PATH = os.path.join(BASE_PATH, 'photo_jpg')
print("No of Monet Images {}".format(len(os.listdir(MONET_PATH))))

print("No of Photo Images {}".format(len(os.listdir(PHOTO_PATH))))
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(1, figsize=(15, 10))

NUM_ROWS = 5

grid = ImageGrid(fig, 111, nrows_ncols=(NUM_ROWS, 5), axes_pad=0.05)

i = 0

imgid = 0

for row_id in range(NUM_ROWS):

    for filepath in os.listdir(MONET_PATH)[imgid:imgid+5]:

        ax = grid[i]

        img = Image.open(MONET_PATH+'/'+filepath)

        img = img.resize((240,240))

        ax.imshow(img)

        ax.axis('off')

        i += 1

        imgid+=5

plt.show();
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(1, figsize=(15, 10))

NUM_ROWS = 5

grid = ImageGrid(fig, 111, nrows_ncols=(NUM_ROWS, 5), axes_pad=0.05)

i = 0

imgid=0



for row_id in range(NUM_ROWS):

    for filepath in os.listdir(PHOTO_PATH)[imgid:imgid+5]:

        ax = grid[i]

        img = Image.open(PHOTO_PATH+'/'+filepath)

        img = img.resize((240,240))

        ax.imshow(img,aspect='auto',extent=(20, 150, 20, 150))

        ax.axis('off')

        i += 1

        imgid+=5

plt.show();