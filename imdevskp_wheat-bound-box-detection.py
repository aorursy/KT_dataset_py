# for file operations

import os



# opencv for computer vision

import cv2

# basic image loading and display operations

import matplotlib.image as mpimg

# ploting

import matplotlib.pyplot as plt





# linear algebra

import numpy as np 

# data processing, CSV file I/O

import pandas as pd 
# list of available files

! ls ../input/global-wheat-detection
# no. of images in train and test folder

print('No. of files in train folder :', len(os.listdir('../input/global-wheat-detection/train')))

print('No. of files in test folder :', len(os.listdir('../input/global-wheat-detection/test')))
# train features and lable

df = pd.read_csv('../input/global-wheat-detection/train.csv')

df.head()
# no. of rows and columns in the dataframe

df.shape
# images with most bounding boxes

df['image_id'].value_counts()[:5]
# Is there any other image size than 1024 ?

print('No. of images with width not equal to 1024 :', df[df['width']!=1024].shape[0])

print('No. of images with height not equal to 1024 :', df[df['height']!=1024].shape[0])
# how many images from each source

df['source'].value_counts()
# df[df['bbox'] == '[, , , ]']
group_by_img_id = df['image_id'].value_counts()

img_wo_bbox = group_by_img_id[group_by_img_id == 1].index

df[df['image_id'].isin(img_wo_bbox)]
# plot single image

# =================



def plot_image(img_id, fig_size=(4, 4)):

    """Plot the image corresponds to the image_id

    

    Args:

        img_src(str) : image id

        fig_size(tuple optional) : figure size     

    """

    

    # complete image path

    img_src = '../input/global-wheat-detection/train/' + img_id + '.jpg'

    # read image

    image = mpimg.imread(img_src)

    # plot image

    plt.figure(figsize=fig_size)

    plt.imshow(image)

    plt.axis('off')

    plt.show()
# plot single image with bounding boxes

# =====================================



def plot_with_bboxes(img_id, fig_size=(4, 4)):

    """Plot the image with bounding box corresponds to the image_id

    

    Args:

        img_src(str) : image id

        fig_size(tuple optional) : figure size     

    """

    

    # complete image path

    img_src = '../input/global-wheat-detection/train/' + img_id + '.jpg'

    # read image

    img = mpimg.imread(img_src)

    # convert image to np array

    img = np.array(img)

    

    # extract bboxes

    bboxes = [eval(i) for i in df[df['image_id'] == '7ad46c7f4']['bbox']]



    # plot

    fig, ax = plt.subplots(1, 1, figsize=fig_size)



    for box in bboxes:



        x = int(box[0])

        y = int(box[1])

        w = int(box[2])

        h = int(box[3])



        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 3)



    ax.set_axis_off()

    ax.imshow(img)
# plot images in subplots

# =======================



def subplot_images(img_id_array, fig_size=(20, 4)):

    """Plot the images in a subplot

    

    Args:

        img_id_array(list) : list of image ids

        fig_size(tuple optional) : figure size     

    """

    

    # path to the dir

    src = '../input/global-wheat-detection/train/'

    # list of image srcs

    img_src = [src + img_id + '.jpg' for img_id in img_id_array]

    

    # plot in subplots

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes = axes.flatten()



    for ind, src in enumerate(img_src[:5]):

        

        # read image

        img = mpimg.imread(img_src[ind])

        # convert image to numpy array

        img = np.array(img)



        axes[ind].set_title(img_ids[ind])

        axes[ind].set_axis_off()

        axes[ind].imshow(img)
plot_image('7ad46c7f4')
plot_with_bboxes('7ad46c7f4')
subplot_images(img_ids[:5])
img_dir = '../input/global-wheat-detection/train/'

img_ids = [img.replace('.jpg', '') for img in os.listdir(img_dir)]

img_src = [img_dir + img for img in os.listdir(img_dir)]

bboxes = [df[df['image_id'] == img_id]['bbox'] for img_id in img_ids]
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

axes = axes.flatten()



for ind, src in enumerate(img_src[:5]):

    

    img = mpimg.imread(img_src[ind])

    img = np.array(img)



    axes[ind].imshow(img)

    axes[ind].set_title(img_ids[ind])

    axes[ind].set_axis_off()

    axes[ind].imshow(img)
fig, axes = plt.subplots(5, 5, figsize=(20, 20))

axes = axes.flatten()



for ind, src in enumerate(img_src[:25]):

    

    img = mpimg.imread(img_src[ind])

    img = np.array(img)



    axes[ind].imshow(img)

    axes[ind].set_title(img_ids[ind])

    axes[ind].set_axis_off()

    axes[ind].imshow(img)
img = mpimg.imread(img_src[2])

img = np.array(img)



fig, ax = plt.subplots(1, 1, figsize=(5, 5))



for box in bboxes[2]:

    

    box = eval(box)

    x = int(box[0])

    y = int(box[1])

    w = int(box[2])

    h = int(box[3])



    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 3)

    

ax.set_axis_off()

ax.imshow(img)
fig, axes = plt.subplots(5, 5, figsize=(20, 20))

axes = axes.flatten()



for ind, src in enumerate(img_src[:25]):

    

    img = mpimg.imread(img_src[ind])

    img = np.array(img)



    for box in bboxes[ind]:



        box = eval(box)

        x = int(box[0])

        y = int(box[1])

        w = int(box[2])

        h = int(box[3])



        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 3)



    axes[ind].set_title(img_ids[ind])

    axes[ind].set_axis_off()

    axes[ind].imshow(img)