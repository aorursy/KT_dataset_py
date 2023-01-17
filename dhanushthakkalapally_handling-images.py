# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import cv2

# Open source computer vision (Opencv)is a library help to work with images 

import numpy as np 



import pandas as pd 



import sklearn as sk 



import matplotlib.pyplot as plt 



import seaborn as sns
# Loading  a image as a grayscale 

image = cv2.imread('/kaggle/input/plane.jpg',cv2.IMREAD_GRAYSCALE)
# Show Image 

plt.imshow(image,cmap='gray')

# in the place of image we can also give 2d or 3d arrays as input then matplot lib will plot the corresponding output

plt.axis('off')

plt.show()
type(image)
print(image.ndim)

image.shape
# Loading image in color 

image_clr = cv2.imread('/kaggle/input/plane.jpg',cv2.IMREAD_COLOR)

print(image_clr.shape)

print('I am a colored image and my dimensions are ',image_clr.ndim,'and I am in BGR format not RGB')
# Converting the bgr image to rgb format to plot in matplotlib 

image_rgb = cv2.cvtColor(image_clr,cv2.COLOR_BGR2RGB)

# above command converts bgr image to rgb

plt.imshow(image_clr)

plt.title('Without converting in to RGB format ')

plt.show()

plt.imshow(image_rgb)

plt.title('after converting in to RGB format ')

plt.show()
image_100X100 = cv2.resize(image,(100,100))

# resizing the image in to 100X100 pixels

plt.imshow(image_100X100,cmap ='gray')

plt.show()

print('I am rescaled and now my size is ',image_100X100.shape)
image_crop = image[:,:2000]

plt.imshow(image_crop,cmap='gray')

plt.show()
# Blurring image 

image_blur = cv2.blur(image,(50,50))

# above I have used 50X50 kernel 

plt.imshow(image_blur,cmap='gray')

plt.show()
# Defining a kernel to sharpening the image 

kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])



# sharpen image 

image_sharp = cv2.filter2D(image,-1,kernel)



plt.imshow(image_sharp,cmap='gray')

plt.show()
image_gray2 = cv2.imread('/kaggle/input/plane_256x256.jpg',cv2.IMREAD_GRAYSCALE)

# calculate the median intensity 

median_intensity = np.median(image_gray2)

# seting thershold to be one of the standard deviation above and below the median

lower_thershold = int((max(0,(1-.33)*median_intensity)))

upper_thershold = int((min(255,(1+.33)*median_intensity)))

#applying canny edge detector 

image_canny = cv2.Canny(image_gray2,lower_thershold,upper_thershold)



#show Image

plt.imshow(image_canny ,cmap='gray')