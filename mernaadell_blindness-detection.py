import os, sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

from tqdm import tqdm

import PIL

from PIL import Image, ImageOps

import cv2

from sklearn.utils import class_weight, shuffle

from keras.losses import binary_crossentropy

from keras.applications.resnet50 import preprocess_input

import keras.backend as K

import tensorflow as tf

from sklearn.metrics import f1_score, fbeta_score

from keras.utils import Sequence

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



WORKERS = 2

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")

IMG_SIZE = 512

NUM_CLASSES = 5

SEED = 77

TRAIN_NUM = 1000 # use 1000 when you just want to explore new idea, use -1 for full train
df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

#print(df_train)

print(df_train.head(10))

print(df_train.describe)

print(df_train.shape)

x = df_train['id_code']

y = df_train['diagnosis']



x, y = shuffle(x, y, random_state=SEED)

print("x data after shuffling")

print(x)

print("y data after shuffling")

print(y)

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15,

                                                      stratify=y, random_state=SEED)

print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

train_y.hist()

valid_y.hist()
%%time

fig = plt.figure(figsize=(25, 16))

# display 10 images from each class

for class_id in sorted(train_y.unique()):

    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(5, random_state=SEED).iterrows()):

        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))



        plt.imshow(image)

        ax.set_title('Label: %d - %d - %s' % (class_id, idx, row['id_code']) )
%%time

fig = plt.figure(figsize=(25, 16))

for class_id in sorted(train_y.unique()):

    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(5, random_state=SEED).iterrows()):

        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         image=cv2.addWeighted ( image, 0 , cv2.GaussianBlur( image , (0 ,0 ) , 10) ,-4 ,128)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))



        plt.imshow(image, cmap='gray')

        ax.set_title('Label: %d-%d-%s' % (class_id, idx, row['id_code']) )

        
dpi = 80 #inch



# path=f"../input/aptos2019-blindness-detection/train_images/5c7ab966a3ee.png" # notice upper part

path=f"../input/aptos2019-blindness-detection/train_images/cd54d022e37d.png" # lower-right, this still looks not so severe, can be class3

image = cv2.imread(path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width = image.shape

print(height, width)



SCALE=2

figsize = (width / float(dpi))/SCALE, (height / float(dpi))/SCALE



fig = plt.figure(figsize=figsize)

plt.imshow(image, cmap='gray')
%%time

fig = plt.figure(figsize=(25, 16))

for class_id in sorted(train_y.unique()):

    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(5, random_state=SEED).iterrows()):

        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , IMG_SIZE/10) ,-4 ,128) # the trick is to add this line



        plt.imshow(image, cmap='gray')

        ax.set_title('Label: %d-%d-%s' % (class_id, idx, row['id_code']) )
def crop_image1(img,tol=7):

    # img is image data

    # tol  is tolerance

        

    mask = img>tol

    return img[np.ix_(mask.any(1),mask.any(0))]



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

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img

# OLD version of image color cropping, use crop_image_from_gray instead

# The above code work only for 1-channel. Here is my simple extension for 3-channels image

def crop_image(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        h,w,_=img.shape

#         print(h,w)

        img1=cv2.resize(crop_image1(img[:,:,0]),(w,h))

        img2=cv2.resize(crop_image1(img[:,:,1]),(w,h))

        img3=cv2.resize(crop_image1(img[:,:,2]),(w,h))

        

#         print(img1.shape,img2.shape,img3.shape)

        img[:,:,0]=img1

        img[:,:,1]=img2

        img[:,:,2]=img3

        return img



'''all of these do not work'''



def crop_image2(image,threshold=5):

    if len(image.shape) == 3:

        flatImage = np.max(image, 2)

    else:

        flatImage = image

    assert len(flatImage.shape) == 2



    rows = np.where(np.max(flatImage, 0) > threshold)[0]

    if rows.size:

        cols = np.where(np.max(flatImage, 1) > threshold)[0]

        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]

    else:

        image = image[:1, :1]



    return image



def crop_image3(image):

    mask = image > 0



    # Coordinates of non-black pixels.

    coords = np.argwhere(mask)



    # Bounding box of non-black pixels.

    x0, y0 = coords.min(axis=0)

    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

    

    # Get the contents of the bounding box.

    cropped = image[x0:x1, y0:y1]

    return cropped



def crop_image4(image):

    _,thresh = cv2.threshold(image,1,255,cv2.THRESH_BINARY)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]

    x,y,w,h = cv2.boundingRect(cnt)

    crop = image[y:y+h,x:x+w]

    return crop



def load_ben_color(path, sigmaX=10):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

        

    return image
%%time



NUM_SAMP=7

fig = plt.figure(figsize=(25, 16))

for class_id in sorted(train_y.unique()):

    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(NUM_SAMP, random_state=SEED).iterrows()):

        ax = fig.add_subplot(5, NUM_SAMP, class_id * NUM_SAMP + i + 1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        image = load_ben_color(path,sigmaX=30)



        plt.imshow(image)

        ax.set_title('%d-%d-%s' % (class_id, idx, row['id_code']) )
# ## try circle crop

def circle_crop(img, sigmaX=10):   

    """

    Create circular crop around image centre    

    """    

    

    img = cv2.imread(img)

    img = crop_image_from_gray(img)    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    height, width, depth = img.shape    

    

    x = int(width/2)

    y = int(height/2)

    r = np.amin((x,y))

    

    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)

    img = crop_image_from_gray(img)

    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)

    return img 
%%time

## try circle crop

NUM_SAMP=7

fig = plt.figure(figsize=(25, 16))

for class_id in sorted(train_y.unique()):

    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(NUM_SAMP, random_state=SEED).iterrows()):

        ax = fig.add_subplot(5, NUM_SAMP, class_id * NUM_SAMP + i + 1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        image = circle_crop(path,sigmaX=30)



        plt.imshow(image)

        ax.set_title('%d-%d-%s' % (class_id, idx, row['id_code']) )
dpi = 80 #inch



# path=f"../input/aptos2019-blindness-detection/train_images/5c7ab966a3ee.png" # notice upper part

path=f"../input/aptos2019-blindness-detection/train_images/cd54d022e37d.png" # lower-right, can be class3

image = load_ben_color(path,sigmaX=10)



height, width = IMG_SIZE, IMG_SIZE

print(height, width)



SCALE=1

figsize = (width / float(dpi))/SCALE, (height / float(dpi))/SCALE



fig = plt.figure(figsize=figsize)

plt.imshow(image, cmap='gray')