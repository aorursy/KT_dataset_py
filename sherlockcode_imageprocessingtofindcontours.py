import numpy as np
from matplotlib import pyplot as plt

import skimage.io
#in order to draw conours lines
from skimage import measure
from skimage import color
from skimage.util import view_as_blocks

shapes = skimage.io.imread("../input/imageastronaut/astronaut.jpg")
shapes.shape

shapes = color.rgb2gray(shapes)
plt.imshow(shapes, cmap= 'gray')

contours = measure.find_contours(shapes,0.01)
len(contours)
#overlaying the contours
fig , axes = plt.subplots(1,2, figsize = (16 , 12), sharex = True , sharey=True)
#Display original image
ax = axes.ravel()
ax[0].imshow(shapes,cmap='gray')
ax[0].set_title('Orignal image', fontsize = 20)
#Display the contour image
ax[1].imshow(shapes,cmap ='gray',interpolation ='nearest')
ax[1].set_title('Contour', fontsize=20)
#enumerate the contours 
for n , contour in enumerate(contours):
    ax[1].plot(contour[:,1], contour[:,0], linewidth=5)
plt.show()

