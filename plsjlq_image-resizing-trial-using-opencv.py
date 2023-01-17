# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm.auto import tqdm 

import time, gc

import cv2



import matplotlib.pyplot as plt





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import warnings

warnings.filterwarnings("ignore")
train = ['/kaggle/input/bengaliai-cv19/train_image_data_0.parquet',

         '/kaggle/input/bengaliai-cv19/train_image_data_1.parquet',

         '/kaggle/input/bengaliai-cv19/train_image_data_2.parquet',

         '/kaggle/input/bengaliai-cv19/train_image_data_3.parquet']



IMG_width = 236

IMG_height = 137

IMG_size = 128
train0 = pd.read_parquet(train[0])
train0.head()
idx = np.random.randint(50410) # set random index



img = train0.iloc[idx, 1:].values

print(f"data type is '{img.dtype}' and item size is {img.itemsize}")

img = train0.iloc[idx, 1:].values.astype(np.uint8)



print(f"data type is '{img.dtype}' and item size is {img.itemsize}")

img = img.reshape(137,236)

plt.imshow(img, cmap='gray')
pd.DataFrame(img)
def img_resize(img, size=IMG_size):

    plt.figure(figsize=(20, 20))

    plt.subplot(4,1,1)

    plt.imshow(img, cmap='gray')

    """

    <In & out>

    input : 137 x 236 size image

    output : IMG_size x IMG_size 

    

    <Applying method>

    # 1) Binary thresholding (cv2.threshold)

    # 2) Resizing with interpolation (cv2.resize)

    # 3) ROI centering (cv2.boundingRect)

    """

    # 1) Binary thresholding (cv2.threshold)

    

    thr = cv2.THRESH_OTSU

    #thr = cv2.THRESH_BINARY

    thr_val, img1 = cv2.threshold(img, 240, 255, thr) # set a threshold : 15

    img1 = 255 - img1

    

    plt.subplot(4,1,2)

    plt.imshow(img1, cmap='gray')

    

    # 2) resizing with interpolation (cv2.resize)

    

    #interpol = cv2.INTER_NEAREST # it preferred for img zoom

    interpol = cv2.INTER_AREA # it preferred for img decimation

    img2 = cv2.resize(img1, (size,size), interpolation=interpol)

        

    plt.subplot(4,1,3)

    plt.imshow(img2, cmap='gray')

    

        

    # 3) ROI centering 

    

    x,y,w,h = cv2.boundingRect(img2) # find bounding box of the character

    ROI = img2[y:y+h, x:x+w]

    img3 = np.zeros((size, size)) # make a canvas for new_img

    center_x, center_y = x + w//2, y + h//2 # find center point of the rectangle

    moving_x, moving_y = (size//2 - center_x), (size//2 - center_y)

    new_x, new_y = x + moving_x, y + moving_y

    

    img3[new_y:new_y+h, new_x:new_x+w] = ROI

        

    plt.subplot(4,1,4)

    plt.imshow(img3, cmap='gray')

    

    new_img = img3.reshape(-1) # reshape 2D image array into 1D array

    new_img = pd.DataFrame(new_img).T # change array into DataFrame 

    

    return new_img
img_resize(img)