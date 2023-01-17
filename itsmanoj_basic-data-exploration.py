# import libraries

import os

import numpy as np

import cv2

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

# create list of all directories

base_dir = '/kaggle/input/alaska2-image-steganalysis/'

image_dirs = ['Cover','JUNIWARD', 'JMiPOD',  'UERD']





all_files = {}

for id in image_dirs:

    if id=="Test":

        continue

    lst = []

    for file in os.listdir(os.path.join(base_dir, id)):

        lst.append(file)

    all_files[id]=lst



files_df = pd.DataFrame(all_files, columns=all_files.keys())

files_df.head(5)




fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))

fig.suptitle("Visual comparison of files")

image_list = []

for d in image_dirs:

    for f in files_df['Cover'].head(4):

        image_list.append(os.path.join(base_dir,d, f))

#print(image_list)



for ax, fname in zip(axes.flatten(), image_list) :

    #print(fname)

    img = mpimg.imread(fname)

    ax.imshow(img)

    ax.set_title(fname.split("/")[-1])



for row, folder_name in zip(axes[:,0], image_dirs):

        row.set_ylabel(folder_name)

plt.show()

sample_image = "41732.jpg"





#image_dirs = ['Cover','JUNIWARD', 'JMiPOD',  'UERD']

img_c = mpimg.imread(os.path.join(base_dir,"Cover",sample_image))

img_u = mpimg.imread(os.path.join(base_dir,"UERD",sample_image))

img_j = mpimg.imread(os.path.join(base_dir,"JUNIWARD",sample_image))

img_jm = mpimg.imread(os.path.join(base_dir,"JMiPOD",sample_image))







fig, ax = plt.subplots(2,2, figsize=(10,10))

fig.suptitle("Differential pixels visualization")

ax[0,0].imshow(img_c)

ax[0,1].imshow((img_c-img_u)*10000)

ax[1,0].imshow((img_c-img_j)*10000)

ax[1,1].imshow((img_c-img_jm)*10000)



ax[0,0].set_title("Original image")

ax[0,1].set_title("UERD differential")

ax[1,0].set_title("JUNIWARD differential")

ax[1,1].set_title("JMiPOD differential")

plt.show()

sample_image = "00014.jpg"





#image_dirs = ['Cover','JUNIWARD', 'JMiPOD',  'UERD']

img_c = mpimg.imread(os.path.join(base_dir,"Cover",sample_image))

img_u = mpimg.imread(os.path.join(base_dir,"UERD",sample_image))

img_j = mpimg.imread(os.path.join(base_dir,"JUNIWARD",sample_image))

img_jm = mpimg.imread(os.path.join(base_dir,"JMiPOD",sample_image))







fig, ax = plt.subplots(2,2, figsize=(10,10))

fig.suptitle("Differential pixels visualization")

ax[0,0].imshow(img_c)

ax[0,1].imshow((img_c-img_u))

ax[1,0].imshow((img_c-img_j))

ax[1,1].imshow((img_c-img_jm))



ax[0,0].set_title("Original image")

ax[0,1].set_title("UERD differential")

ax[1,0].set_title("JUNIWARD differential")

ax[1,1].set_title("JMiPOD differential")

plt.show()

sample_image = "12314.jpg"





#image_dirs = ['Cover','JUNIWARD', 'JMiPOD',  'UERD']

img_c = mpimg.imread(os.path.join(base_dir,"Cover",sample_image))

img_u = mpimg.imread(os.path.join(base_dir,"UERD",sample_image))

img_j = mpimg.imread(os.path.join(base_dir,"JUNIWARD",sample_image))

img_jm = mpimg.imread(os.path.join(base_dir,"JMiPOD",sample_image))







fig, ax = plt.subplots(2,2, figsize=(10,10))

fig.suptitle("Differential pixels visualization")

ax[0,0].imshow(img_c)

ax[0,1].imshow((img_c-img_u))

ax[1,0].imshow((img_c-img_j))

ax[1,1].imshow((img_c-img_jm))



ax[0,0].set_title("Original image")

ax[0,1].set_title("UERD differential")

ax[1,0].set_title("JUNIWARD differential")

ax[1,1].set_title("JMiPOD differential")

plt.show()