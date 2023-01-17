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
import numpy as np

import cv2

import matplotlib.pyplot as plt

from PIL import Image 
pic=Image.open('..//input/L1_out.png')
pic
type(pic)
pic_arr=np.asarray(pic)
type(pic_arr)
pic_arr.shape
plt.imshow(pic_arr)

plt.ioff()
pic_red=pic_arr.copy()
plt.imshow(pic_red)

plt.ioff()
pic_red.shape
# R G B 

plt.imshow(pic_red[:,:,0])   # O Stands for Red

plt.ioff()
plt.imshow(pic_red[:,:,1])    # 1 stands for Green

plt.ioff()
plt.imshow(pic_red[:,:,2])   # 2 stands for blue

plt.ioff() 
# RED Channel value 0 means no red,Pure black 

#value of 255 means full read and will appear white in image



plt.imshow(pic_red[:,:,0],cmap='gray')   # 2 stands for blue

plt.ioff()
plt.imshow(pic_red[:,:,1],cmap='gray')   # 2 stands for blue

plt.ioff()
plt.imshow(pic_red[:,:,2],cmap='gray')   # 2 stands for blue

plt.ioff()
# Making the green channel zero 

pic_red[:,:,1]=0
# Making the blue channel zero 

pic_red[:,:,2]=0
plt.imshow(pic_red)

plt.ioff()
img=cv2.imread('..//input/L1_out.png')
type(img)
img.shape
plt.imshow(img)

plt.ioff()
fix_image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(fix_image)

plt.ioff()
img_gray=cv2.imread('..//input/L1_out.png',cv2.IMREAD_GRAYSCALE)
img_gray.max()
plt.imshow(img_gray)

plt.ioff()
plt.imshow(img_gray,cmap='gray')

plt.ioff()
plt.imshow(img_gray,cmap='magma')

plt.ioff()
fix_image.shape
new_image=cv2.resize(fix_image,(1000,500))
plt.imshow(new_image)

plt.ioff()
w_ratio=0.5

h_ratio=0.5
new_image_1=cv2.resize(fix_image,(0,0),fix_image,w_ratio,h_ratio)
plt.imshow(new_image_1)

plt.ioff()
new_image_1.shape
new_img_f=cv2.flip(fix_image,0)

plt.imshow(new_img_f)

plt.ioff()
new_img_f=cv2.flip(fix_image,1)

plt.imshow(new_img_f)

plt.ioff()
new_img_f=cv2.flip(fix_image,-1)

plt.imshow(new_img_f)

plt.ioff()
#cv2.imwrite('',fix_image)
fig=plt.figure(figsize=(10,8))

ax=fig.add_subplot(111)

ax.imshow(fix_image)

plt.ioff()