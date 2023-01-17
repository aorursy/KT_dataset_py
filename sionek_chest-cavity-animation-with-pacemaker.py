import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import dicom

import os

import pylab

import cv2



images_path = '../input/sample_images/'
#r4m0n

def get_3d_data(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    return np.stack([s.pixel_array for s in slices])
#r4m0n

patients = os.listdir(images_path)
patients
sample_image = get_3d_data(images_path + patients[1])

sample_image.shape
#r4m0n

#the images have the unavailable pixel set to -2000, changing them to 0 makes the picture clearer

sample_image[sample_image == -2000] = 0

#remaping the image to 1 standard deviation of the average and clipping it to 0-1

#img_std = np.std(sample_image)

#img_avg = np.average(sample_image)

#std_image = np.clip((sample_image - img_avg + img_std) / (img_std * 2), 0, 1)
#r4m0n

#same plane as the original data, cut at the Z axis

for i in range(27,28):

    pylab.imshow(sample_image[i], cmap=pylab.cm.bone)

  

    pylab.show()

    
#sample_image = std_image



%matplotlib nbagg

import matplotlib.animation as animation

fig = plt.figure() # make figure

from IPython.display import HTML



im = plt.imshow(sample_image[1], cmap=pylab.cm.bone)



# function to update figure

def updatefig(j):

    # set the data in the axesimage object

    im.set_array(sample_image[j])

    # return the artists set

    return im,

# kick off the animation

ani = animation.FuncAnimation(fig, updatefig, frames=range(len(sample_image)), 

                              interval=100, blit=True)

ani.save('Chest_Cavity2.gif', writer='imagemagick')

plt.show()