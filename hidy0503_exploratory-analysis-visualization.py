import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import dicom

import cv2

from skimage import data, io, filters

import os

import matplotlib.pyplot as plt

import pylab



os.chdir("../input/sample_images")

files = os.listdir()



%config InlineBackend.figure_format = 'retina'
def show(slice):

    plt.imshow(slice, cmap=plt.cm.bone)
os.chdir('0d941a3ad6c889ac451caf89c46cb92a')
os.listdir()[:5]
files = os.listdir()



imgs = []

for i in files:

        ds = dicom.read_file(i)

        print(ds)

        imgs.append(ds)
#sorting based on InstanceNumber stolen from r4m0n's script: 

imgs.sort(key = lambda x: int(x.InstanceNumber))

full_img = np.stack([s.pixel_array for s in imgs])
print(full_img)

print(full_img.shape)

full_img.shape
print(full_img[0])

print(full_img[0].shape)

print(len(full_img))
#from https://www.kaggle.com/z0mbie/data-science-bowl-2017/chest-cavity-animation-with-pacemaker

#not working..hmm

%matplotlib nbagg

import matplotlib.animation as animation

fig = plt.figure() # make figure

from IPython.display import HTML



im = plt.imshow(full_img[0], cmap=pylab.cm.bone)



# function to update figure

def updatefig(j):

    # set the data in the axesimage object

    im.set_array(full_img[j])

    # return the artists set

    return im,

# kick off the animation

ani = animation.FuncAnimation(fig, updatefig, frames=range(len(full_img)), 

                              interval=50, blit=True)

ani.save('Chest_Cavity.gif', writer='imagemagick')

plt.show()
%matplotlib inline

matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)





for i in range(36):

    plt.subplot(6,6,i+1)

    show(full_img[4*i,:,:])    

    plt.xticks([])

    plt.yticks([])
print(full_img[175,:,:])



show(full_img[175])
for i in range(36):

    plt.subplot(6,6,i+1)

    img = cv2.resize(full_img[:,20+ 12*i,:], (256, 256))

    show(img)    

    plt.xticks([])

    plt.yticks([])
for i in range(36):

    plt.subplot(6,6,i+1)

    img = cv2.resize(full_img[:,:,20+ 12*i], (256, 256))

    show(img)    

    plt.xticks([])

    plt.yticks([])
full_img[100]
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



pd.Series(full_img[10].ravel()).hist(bins = 100)
edge_img = filters.sobel(full_img[90,:,:]/2500)
plt.imshow(edge_img)
plt.imshow(edge_img > 0.04, cmap = "viridis")
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)





for i in range(36):

    plt.subplot(6,6,i+1)

    edge_img = filters.sobel(full_img[4*i,:,:]/2500)

    img = edge_img > 0.04

    plt.imshow(img, cmap = "viridis")    

    plt.xticks([])

    plt.yticks([])