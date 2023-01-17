import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt  

%matplotlib inline

from skimage import color,viewer,transform,feature,data,filters,io

from skimage.filters import edges

from PIL import Image
def bg_color_changer(image,bgcolor=[255,255,255, 0.3]):

    x = image

    h = len(image)

    w = len(image[0])

    x_flat = pd.Series(list(zip(*[iter(x.flatten())] * 4)))

    y = x_flat.value_counts().idxmax()

    for a in range(h):

        for b in range(w):

            if all(x[a][b] == y):

                x[a][b] = bgcolor

    plt.imshow(x)
x1 = io.imread('../input/image-altering-dataset/x1.png')

x2 = io.imread('../input/image-altering-dataset/x2.png')

y1 = io.imread('../input/image-altering-dataset/y1.png')

test = io.imread('../input/image-altering-dataset/test.png')

y2 = io.imread('../input/image-altering-dataset/y2.png')

X = np.array(Image.open('../input/image-altering-dataset/x1.png'), dtype=np.uint8)
plt.imshow(x1)
bg_color_changer(x1)
bg_color_changer(x1,bgcolor=[0,255,255,255])
bg_color_changer(x1,[255,0,255,255])
bg_color_changer(x1,bgcolor=[255,255,0,255])
from skimage.data import camera

from skimage import feature

from skimage import segmentation
cheetah = io.imread('../input/mask-dataset/Cheetah.png')
plt.imshow(filters.edges.prewitt(color.rgb2gray(x2)))
plt.imshow(filters.sobel(color.rgb2gray(cheetah)))
plt.imshow(feature.canny(color.rgb2gray(x1), sigma=3))
plt.imshow(segmentation.morphological_chan_vese(filters.roberts(color.rgb2grey(cheetah)),1))
plt.imshow(segmentation.morphological_chan_vese((color.rgb2gray(cheetah)),1))
plt.imshow(segmentation.chan_vese(filters.roberts(color.rgb2gray(cheetah))))
plt.imshow(feature.canny(color.rgb2gray(cheetah),sigma=3))