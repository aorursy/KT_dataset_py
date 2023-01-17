# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from IPython.display import display_html

from PIL import Image

import gc

import cv2

import pydicom # for DICOM images

from skimage.transform import resize



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Any results you write to the current directory are saved as output.

image = cv2.imread('/kaggle/input/totaltextstr/Total-Text/Test/img1093.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(12, 12))

plt.subplot(1, 2, 1)

plt.title("Original")

plt.imshow(image)



# Create our shapening kernel, we don't normalize since the 

# the values in the matrix sum to 1

kernel_sharpening = np.array([[-1,-1,-1], 

                              [-1,9,-1], 

                              [-1,-1,-1]])



# applying different kernels to the input image

sharpened = cv2.filter2D(image, -1, kernel_sharpening)





plt.subplot(1, 2, 2)

plt.title("Image Sharpening")

plt.imshow(sharpened)



plt.show()
#Codes from Andrada Olteanu https://www.kaggle.com/andradaolteanu/siim-melanoma-competition-eda-augmentations/notebook¶

#for i in range(0, 2*6):

  #  data = pydicom.read_file(df['input/bird-species/dataset1.0/train/Hawk/09bf87799e7b43308a0c5bdff428ba84.jpg'][i])

   # image = data.pixel_array

image = cv2.imread('/kaggle/input/totaltextstr/Total-Text/Test/img1093.jpg')    

    # Transform to B&W

    # The function converts an input image from one color space to another.

image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

image = cv2.resize(image, (200,200))

    

#x = i // 6

#y = i % 6

#axes[x, y].imshow(image, cmap=plt.cm.bone) 

#axes[x, y].axis('off')

plt.imshow(image)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16,5))



#for i in range(0, 2*6):

 #   data = pydicom.read_file(train_df['path_ditotaltextstr/Total-Text/Test/img1093com'][i])

 #   image = data.pixel_array

image = cv2.imread('/kaggle/input/totaltextstr/Total-Text/Test/img1093.jpg')     

    # Transform to B&W

    # The function converts an input image from one color space to another.

image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

image = cv2.resize(image, (200,200))

    

#    x = i // 6

 #   y = i % 6

  #  axes[x, y].imshow(image, cmap=plt.cm.bone) 

   # axes[x, y].axis('off')

plt.imshow(image) 