# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
import numpy as np

from matplotlib import pyplot as plt

image=cv2.imread("../input/natural-images/data/natural_images/fruit/fruit_0020.jpg")
img=cv2.imread("../input/natural-images/data/natural_images/fruit/fruit_0089.jpg")
image
plt.imshow(image,cmap="gray"),plt.axis("off")

plt.show()
image.shape
image[0,0]
# Load libraries 

import cv2 

import numpy as np

from matplotlib import pyplot as plt

# Load image as grayscale 

image=cv2.imread("../input/natural-images/data/natural_images/fruit/fruit_0020.jpg",cv2.IMREAD_GRAYSCALE)

# Save image 

cv2.imwrite("../input/output.jpg", image)

image=cv2.imread("../input/natural-images/data/natural_images/fruit/fruit_0020.jpg",cv2.IMREAD_GRAYSCALE)

img_50X50=cv2.resize(image,(20,20))

plt.imshow(img_50X50,cmap="gray"),plt.axis("off")

plt.show()
plt.imshow(image,cmap="gray"),plt.axis("off")
image_cropped=image[:,:98]

plt.imshow(image_cropped,cmap="gray"),plt.axis("off")
image_blurry=cv2.blur(image,(5,5))

plt.imshow(image_blurry,cmap="gray"),plt.axis("off")
image_very_blurry=cv2.blur(image,(100,100))

plt.imshow(image_very_blurry,cmap="gray"),plt.axis("off")
kernel=np.ones((5,5))/25.0

kernel
image_kernal=cv2.filter2D(image,-1,kernel)

plt.imshow(image_kernal,cmap="gray"),plt.xticks([]),plt.yticks([])

kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

image_sharp=cv2.filter2D(image,-1,kernel)

plt.imshow(image_sharp,cmap="gray"),plt.axis("off")

plt.show()
image_enhanced=cv2.equalizeHist(image)

plt.imshow(image_enhanced,cmap="gray"),plt.axis("off")

plt.show()
image_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])

image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

plt.imshow(image_rgb), plt.axis("off") 

plt.show()
