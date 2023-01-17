# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
img = Image.open('../input/rgb-bw/RGB_BW.jpg')
plt.imshow(img)
img_arr = np.asarray(img)
img_arr.shape 
plt.imshow(img_arr[:,:,0],cmap='gray')
#for only green color
img_green = img_arr.copy()
img_green[:,:,0] = 0 #red pixels = 0
img_green[:,:,2] = 0 #green pixels = 0

plt.imshow(img_green) #why does white turn into green ?? (because white is made up of RGB)

import cv2
pic = cv2.imread('../input/koala-/Koala.jpg')
type(pic)   #because of cv2 we dont need to convert it to array, it has already been read as array
plt.imshow(pic) #color inversion RGB<->BGR
pic.shape   
original_img = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
plt.imshow(original_img)
original_img.shape
resized_img = cv2.resize(original_img,(500,250)) #X-Y are taken in opposite manner over here
plt.imshow(resized_img)
#reducing by ratio 
W_RATIO = 0.3
L_RATIO = 0.3
ratio_img = cv2.resize(original_img,(0,0),original_img,W_RATIO,L_RATIO)
plt.imshow(ratio_img)
flipped_img_H = cv2.flip(original_img,0)
plt.imshow(flipped_img_H)
flipped_img_V = cv2.flip(original_img,1)
plt.imshow(flipped_img_V)
flipped_img_HV = cv2.flip(original_img,-1)
plt.imshow(flipped_img_HV)
cv2.imwrite('flipped_img.jpg',flipped_img_HV)
