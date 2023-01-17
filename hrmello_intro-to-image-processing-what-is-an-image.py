import numpy as np # linear algebra

import cv2 # OpenCV module that handles computer vision methods

import matplotlib.pyplot as plt # make plots

import os



ANNOTATION_DIR = '../input/annotations/Annotation/'

IMAGES_DIR = '../input/images/Images/'



# list of breeds of dogs in the dataset

breed_list = os.listdir(ANNOTATION_DIR)



## set the seed for the np.random module, so we always get the same image when run this code cell

np.random.seed(35)



# since we just want one image, I'll ramdomly choose a breed and a dog from that breed

breed = np.random.choice(breed_list)

dog = np.random.choice(os.listdir(ANNOTATION_DIR + breed))



# opening one image

img = cv2.imread(IMAGES_DIR + breed + '/' + dog + '.jpg') 



# this line is necessary because cv2 reads an image in BGR format (Blue, Green, Red) by default. 

# So we will convert it to RGB

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 



plt.figure(figsize = (10,10))

plt.imshow(img);
f, axes = plt.subplots(1,3, figsize = (15,15))

i = 0

colors = {'0':'red', '1': 'green', '2':'blue'}

for ax in axes:

    ax.imshow(img[:,:,i], cmap = "gray")

    i+=1