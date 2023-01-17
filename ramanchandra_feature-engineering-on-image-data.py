import pandas as pd

import numpy as np

import skimage

import matplotlib.pyplot as plt

from skimage import io





cat=io.imread("../input/cat-image/cat.jpg")

dog=io.imread("../input/dog-image/dog.jpeg")
df=pd.DataFrame(['Cat','Dog'],columns=['Image'])

df
print(cat.shape,dog.shape)
fig=plt.figure(figsize=(8,4))

ax1=fig.add_subplot(1,2,1)

ax2=fig.add_subplot(1,2,2)

ax1.imshow(cat)

ax2.imshow(dog)


dog_r=dog.copy() #REd Channel

dog_r[:,:,1]=dog_r[:,:,2]=0  #Set G,B pixels=0

dog_g=dog.copy() # Green Channel

dog_g[:,:,0]=dog_g[:,:,2]=0  #Set R,B pixels=0

dog_b=dog.copy() # Blue Channel

dog_b[:,:,0]=dog_b[:,:,1]=0 # #Set R,G pixels=0

plot_image=np.concatenate((dog_r,dog_g,dog_b),axis=1)

plt.figure(figsize=(10,4))

plt.imshow(plot_image)           
cat_r=cat.copy() #REd Channel

cat_r[:,:,1]=cat_r[:,:,2]=0  #Set G,B pixels=0

cat_g=cat.copy() # Green Channel

cat_g[:,:,0]=cat_g[:,:,2]=0  #Set R,B pixels=0

cat_b=cat.copy() # Blue Channel

cat_b[:,:,0]=cat_b[:,:,1]=0 # #Set R,G pixels=0

plot_image=np.concatenate((cat_r,cat_g,cat_b),axis=1)

plt.figure(figsize=(10,4))

plt.imshow(plot_image)     
from skimage.color import rgb2gray

cgs=rgb2gray(cat)

dgs=rgb2gray(dog)

print('Image Shape:\n',cat.shape,dog.shape)
print('2D pixel map for cat')

print(np.round(cgs,2))
print('Flattened pixel map:',(np.round(cgs.flatten(),2)))
fig=plt.figure(figsize=(8,4))

ax1=fig.add_subplot(2,2,1)

ax1.imshow(cgs,cmap='gray')

ax2=fig.add_subplot(2,2,2)

ax2.imshow(dgs,cmap='gray')

ax3=fig.add_subplot(2,2,3)

c_freq,c_bins,c_patches=ax3.hist(cgs.flatten(),bins=30)

ax4=fig.add_subplot(2,2,4)

d_freq,d_bins,d_patches=ax4.hist(dgs.flatten(),bins=30)
from skimage.feature import canny

cat_edges=canny(cgs,sigma=2.5)

dog_edges=canny(dgs,sigma=2.5)



fig=plt.figure(figsize=(10,6))

ax1=fig.add_subplot(1,2,1)

ax1.imshow(cat_edges,cmap='binary')

ax2=fig.add_subplot(1,2,2)

ax2.imshow(dog_edges,cmap='binary')
from skimage.feature import hog

from skimage import exposure



fd_cat,cat_hog = hog(cgs,orientations=8,pixels_per_cell=(8,8),cells_per_block=(3,3),visualise=True)

fd_dog,dog_hog = hog(dgs,orientations=8,pixels_per_cell=(8,8),cells_per_block=(3,3),visualise=True)



cat_hogs=exposure.rescale_intensity(cat_hog,in_range=(0,0.04))

dog_hogs=exposure.rescale_intensity(dog_hog,in_range=(0,0.04))

fig=plt.figure(figsize=(10,4))

ax1=fig.add_subplot(1,2,1)

ax1.imshow(cat_hogs,cmap='binary')

ax2=fig.add_subplot(1,2,2)

ax2.imshow(dog_hogs,cmap='binary')
print("Flattened feature vector for cat image:\n",fd_cat,fd_cat.shape)

print("Flattened feature vector for dog image:\n",fd_dog,fd_dog.shape)