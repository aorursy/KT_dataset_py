import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

from tqdm import tqdm

import cv2
IM_DIM = 224
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
def get_pad_width(im, new_shape, is_rgb=True):

    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]

    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)

    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)

    if is_rgb:

        pad_width = ((t,b), (l,r), (0, 0))

    else:

        pad_width = ((t,b), (l,r))

    return pad_width



def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

      

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img



def clahe_green(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=0.5)

    lab_planes[1] = clahe.apply(lab_planes[1])

    lab = cv2.merge(lab_planes)

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return image



def reshape_img(image, im_dim):

    image = cv2.resize(image, (im_dim,)*2)

    return image



def load_ben_color(image, sigmaX):

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

    return image



def preprocess(image):

    image = crop_image_from_gray(image)

    image = clahe_green(image)

    image = reshape_img(image, im_dim = IM_DIM)

    image = load_ben_color(image, sigmaX=30)

    return image
NUM_SAMP=5

SEED = 7

fig = plt.figure(figsize=(25, 16))

for class_id in sorted(train_df['diagnosis'].unique()):

    for i, (idx, row) in enumerate(train_df.loc[train_df['diagnosis'] == class_id].sample(NUM_SAMP, random_state=SEED).iterrows()):

        ax = fig.add_subplot(5, NUM_SAMP, class_id * NUM_SAMP + i + 1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        image = cv2.imread(path)

        image = preprocess(image)

        plt.imshow(image)

        ax.set_title('%d-%d-%s' % (class_id, idx, row['id_code']) )
image.shape
def img_arr(df, im_dim, path):

    N = df.shape[0]

    arr = np.empty((N, im_dim, im_dim, 3), dtype = np.uint8)

    for i, image_id in enumerate(tqdm(df['id_code'])):

        image = cv2.imread(f'{path}/{image_id}.png')

        arr[i,:,:,:] = preprocess(image)

    return arr
x_train = img_arr(train_df, im_dim = IM_DIM, path = '../input/aptos2019-blindness-detection/train_images/')
x_test = img_arr(test_df, im_dim = IM_DIM, path = '../input/aptos2019-blindness-detection/test_images/')
np.save('train_all_four',x_train)

np.save('test_all_four',x_test)