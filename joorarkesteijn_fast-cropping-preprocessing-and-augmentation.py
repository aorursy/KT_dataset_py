import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

IMAGE_SIZE = 224
def info_image(im):

    # Compute the center (cx, cy) and radius of the eye

    cy = im.shape[0]//2

    midline = im[cy,:]

    midline = np.where(midline>midline.mean()/3)[0]

    if len(midline)>im.shape[1]//2:

        x_start, x_end = np.min(midline), np.max(midline)

    else: # This actually rarely happens p~1/10000

        x_start, x_end = im.shape[1]//10, 9*im.shape[1]//10

    cx = (x_start + x_end)/2

    r = (x_end - x_start)/2

    return cx, cy, r



def resize_image(im, augmentation=True):

    # Crops, resizes and potentially augments the image to IMAGE_SIZE

    cx, cy, r = info_image(im)

    scaling = IMAGE_SIZE/(2*r)

    rotation = 0

    if augmentation:

        scaling *= 1 + 0.3 * (np.random.rand()-0.5)

        rotation = 360 * np.random.rand()

    M = cv2.getRotationMatrix2D((cx,cy), rotation, scaling)

    M[0,2] -= cx - IMAGE_SIZE/2

    M[1,2] -= cy - IMAGE_SIZE/2

    return cv2.warpAffine(im,M,(IMAGE_SIZE,IMAGE_SIZE)) # This is the most important line



def subtract_median_bg_image(im):

    k = np.max(im.shape)//20*2+1

    bg = cv2.medianBlur(im, k)

    return cv2.addWeighted (im, 4, bg, -4, 128)



def subtract_gaussian_bg_image(im):

    k = np.max(im.shape)/10

    bg = cv2.GaussianBlur(im ,(0,0) ,k)

    return cv2.addWeighted (im, 4, bg, -4, 128)



def id_to_image(id_code, resize=True, augmentation=False, subtract_gaussian=False, subtract_median=False):

    path = '../input/train_images/{}.png'.format(id_code)

    im = cv2.imread(path)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if resize_image:

        im = resize_image(im, augmentation)

    if subtract_gaussian:

        im = subtract_gaussian_bg_image(im)

    if subtract_median:

        im = subtract_median_bg_image(im)

    return im
%%time

df_train = pd.read_csv('../input/train.csv')

fig = plt.figure(figsize=(25, 16))

SEED = np.random.randint(0,100)



def plot_col(col, id2im, n_cols=6):

    for class_id in range(0,5):

        for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(1, random_state=SEED).iterrows()):

            ax = fig.add_subplot(5, n_cols, class_id * n_cols + i + col, xticks=[], yticks=[])

            im = id2im(row['id_code'])

            plt.imshow(im)

            ax.set_title('Label: %d-%d-%s' % (class_id, idx, row['id_code']) )



# display normal image of each class

plot_col(1, lambda x: id_to_image(x))



# display normal image of each class

plot_col(4, lambda x: id_to_image(x, augmentation=True))



# display normal image of each class

plot_col(2, lambda x: id_to_image(x, subtract_gaussian=True))



# display normal image of each class

plot_col(5, lambda x: id_to_image(x, subtract_gaussian=True, augmentation=True))



# display normal image of each class

plot_col(3, lambda x: id_to_image(x, subtract_median=True))



# display normal image of each class

plot_col(6, lambda x: id_to_image(x, subtract_median=True, augmentation=True))