# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import math

import os

import matplotlib.pyplot as plt

import numpy as np

from skimage.transform import resize

from skimage.transform import rotate

from skimage import color

from glob import glob
#Custom directories

custom_files = glob('data/RawFeatures/*.*') #Input Dir for my cropped images

samples_dir = 'data/classes/' #Output directory for cropped images

samples = glob('data/classes/*.*') # input 

augmented = 'data/augmented/' # folder to store augmented samples

#

training_dotted = glob('../input/TrainDotted/*.jpg')

trainset = glob('../input/Train/*.jpg')
#Misc data structures 

classes = ['adult_males','subadult_males','adult_females','juveniles','pups','error']

tel = {'red': 1, 'magenta': 1,'brown':1,'blue':1,'green':1,'black':0}

keys = ['red', 'magenta','brown','blue','green','black']
#Input Image

#output cropped images based on the specified dimensions



def squared_slice(image_path, out_name, outdir, slice_size,threshold):

    from skimage.io import imsave

    img = plt.imread(image_path)

    #---------------------------------------------------------

    img=img[0:500,1000:1500] #remove this line before using

    #---------------------------------------------------------

    width, height, d= img.shape



    slices_h = int(math.ceil(height/slice_size))

    slices_v = int(math.ceil(width/slice_size))

    name = image_path # (image_path.split("\\")[1]).split(".")[0]



    if width*height > threshold:

        n=0

        for s_h in range(slices_h-1):

            for s_v in range(slices_v-1):

                x_r1 = ((s_h)*slice_size)

                x_r2 = (slice_size*s_h+slice_size)

                

                x_y1 = ((s_v)*slice_size)

                x_y2 = (slice_size*s_v+slice_size)

                



                slic = img[x_r1:x_r2,x_y1:x_y2] #Cropped

                plt.imshow(slic)

                plt.show()

                # I save the files in a different folder for further processing

                try:

                    filename = samples_dir + out_name +"-" + name+"-" +"-" +str(n) +".jpg"

               #     imsave(filename, slic)

                except:

                     print( image_path +":" + str(x_r1)+str(x_r2) + str(x_y1) + str(x_y2) )

                n+=1

    else:

        filename = samples_dir + out_name +"-" + name+"-" +"-" +str(n) +".jpg"

        imsave(filename, slic)

        return
#Input: Image

#Output: One image per rotation angle

# dark areas are pesent around the edges of rotated images.



def rotate_img(img,h,w,name):

    from skimage.io import imsave

    temp=np.zeros((0,h*w))

    img = resize(img, (h,w),mode='reflect')

    img = color.rgb2grey(img)



    #temp = np.append(temp,org,axis=0)

    rads = [ 20,280 ] #increase as needed.



    n=0

    for i in rads:

        img_r = rotate(img,i)



        try:

            plt.imshow(img_r)

            plt.show()

            #imsave( filename , img_r)

        except:

            print("low contrast")

        n+=1

    
#3for i in training_dotted[1]:

#i = i[0:200,0:200]



img = plt.imread(training_dotted[0])

img2=img[0:1000,1000:2000]

print("Original Image")

plt.imshow(img)

plt.show()

    #Y , X

print("cropped images") #

plt.imshow(img[0:1000,1000:2000])

plt.show()

# Arguments

#file path

# output name

# Output dir

# slice size

# Threshold = Width * height

squared_slice(training_dotted[0],'sliced',samples_dir,500,400)
h=500

w=500

img = plt.imread(training_dotted[1])

rotate_img(img,h,w,'test' )