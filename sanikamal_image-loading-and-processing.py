import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Used to change filepaths

from pathlib import Path

import matplotlib.pyplot as plt

import IPython

from IPython.display import display

from PIL import Image

%matplotlib inline

import os

print(os.listdir("../input"))
IPython.display.Image(filename='../input/flowers/flowers/rose/12202373204_34fb07205b.jpg') 
IPython.display.Image(filename='../input/flowers/flowers/sunflower/1008566138_6927679c8a.jpg') 
# generate test_data.

test_data=np.random.beta(1, 1, size=(120, 120, 3))



# display the test_data

plt.imshow(test_data)
# open the image

img = Image.open('../input/flowers/flowers/rose/14510185271_b5d75dd98e_n.jpg')

# Get the image size

img_size = img.size



print("The image size is: {}".format(img_size))



# Just having the image as the last line in the cell will display it in the notebook

img
# Crop the image to 25, 25, 75, 75

img_cropped = img.crop([25,25,75,75])

display(img_cropped)



# rotate the image by 45 degrees

img_rotated = img.rotate(45,expand=25)

display(img_rotated)



# flip the image left to right

img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT) 

display(img_flipped)
# Turn our image object into a NumPy array

img_data = np.array(img)



# get the shape of the resulting array

img_data_shape = img_data.shape



print("Our NumPy array has the shape: {}".format(img_data_shape))



# plot the data with `imshow` 

plt.imshow(img_data)

plt.show()



# plot the red channel

plt.imshow(img_data[:,:,0], cmap=plt.cm.Reds_r)

plt.show()



# plot the green channel

plt.imshow(img_data[:,:,1], cmap=plt.cm.Greens_r)

plt.show()



# plot the blue channel

plt.imshow(img_data[:,:,2], cmap=plt.cm.Blues_r)

plt.show()
def plot_kde(channel, color):

    """ Plots a kernel density estimate for the given data.

        

        `channel` must be a 2d array

        `color` must be a color string, e.g. 'r', 'g', or 'b'

    """

    data = channel.flatten()

    return pd.Series(data).plot.density(c=color)



# create the list of channels

channels = ['r','g','b']

    

def plot_rgb(image_data):

    # use enumerate to loop over colors and indexes

    for ix, color in enumerate(channels):

        plt.imshow(image_data[:,:,ix])

        plt.show()

    

plot_rgb(img_data)
# load rose

rose =Image.open('../input/flowers/flowers/rose/16001846141_393fdb887e_n.jpg')

# display the rose image

display(rose)



# NumPy array of the rose image data

rose_data=np.array(rose)

# plot the rgb densities for the rose image

plot_rgb(rose_data)
# load sunflower

sunflower =Image.open('../input/flowers/flowers/sunflower/1044296388_912143e1d4.jpg')

# display the sunflower image

display(sunflower)

# NumPy array of the sunflower image data

sunflower_data=np.array(sunflower)

# plot the rgb densities for the sunflower image

plot_rgb(sunflower_data)
# convert rose to grayscale

rose_bw = rose.convert("L")

display(rose_bw)



# convert the image to a NumPy array

rose_bw_arr = np.array(rose_bw)



# get the shape of the resulting array

rose_bw_arr_shape = rose_bw_arr.shape

print("Our NumPy array has the shape: {}".format(rose_bw_arr_shape))



# plot the array using matplotlib

plt.imshow(rose_bw_arr, cmap=plt.cm.gray)

plt.show()



# plot the kde of the new black and white array

plot_kde(rose_bw_arr, 'k')
# flip the image left-right with transpose

rose_bw_flip = rose.transpose(Image.FLIP_LEFT_RIGHT)



# show the flipped image

display(rose_bw_flip)



# save the flipped image

rose_bw_flip.save("bw_flipped.jpg")



# create higher contrast by reducing range

rose_hc_arr = np.maximum(rose_bw_arr, 100)



# show the higher contrast version

plt.imshow(rose_hc_arr, cmap=plt.cm.gray)



# convert the NumPy array of high contrast to an Image

rose_bw_hc = Image.fromarray(rose_hc_arr,"L")



# save the high contrast version

rose_bw_hc.save("bw_hc.jpg")
# take only four image from sunflower

image_paths = ['../input/flowers/flowers/sunflower/1022552036_67d33d5bd8_n.jpg',

               '../input/flowers/flowers/sunflower/14121915990_4b76718077_m.jpg',

               '../input/flowers/flowers/sunflower/1043442695_4556c4c13d_n.jpg',

               '../input/flowers/flowers/sunflower/14472246629_72373111e6_m.jpg']



def process_image(path):

    img = Image.open(path)



    # create paths to save files to

    bw_path = "bw_{}.jpg".format(path.stem)

    rcz_path = "rcz_{}.jpg".format(path.stem)



    print("Creating grayscale version of {} and saving to {}.".format(path, bw_path))

    bw = img.convert("L").save(bw_path)

    print("Creating rotated, cropped, and zoomed version of {} and saving to {}.".format(path, bw_path))

    rcz = img.rotate(45).crop([25, 25, 75, 75]).resize((100, 100)).save(rcz_path)



# for loop over image paths

for img_path in image_paths:

    process_image(Path(img_path))