import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from skimage.io import imread, imshow



image = imread('../input/images/index_1.png', as_gray=True)

#image = imread('../input/dog_image/Screenshot_2.png', as_gray=True)

imshow(image)


#checking image shape 

print('Shape of the image is = ',image.shape)



# image matrix

print('\n\nImage matrix\n\n',image)
#*************Grayscale Pixel Values as Features**************

# create features

features = np.reshape(image, (283*280))



# shape of feature array

print('\n\nShape of the feature array = ',features.shape)



print('\n\nFeature Array\n\n',features)



#We will get our feature – which is a 1D array of length 79240.
#***************************** Extracting Edge Features *****************



#importing the required libraries

from skimage.io import imread, imshow

from skimage.filters import prewitt_h,prewitt_v

%matplotlib inline







#calculating horizontal edges using prewitt kernel

edges_prewitt_horizontal = prewitt_h(image)

#calculating vertical edges using prewitt kernel

edges_prewitt_vertical = prewitt_v(image)



imshow(edges_prewitt_vertical, cmap='gray')