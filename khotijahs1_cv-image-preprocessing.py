import datetime as dt

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

sns.set_style('whitegrid')





import os

from keras.applications import xception

from keras.preprocessing import image

from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



import cv2

from scipy.stats import uniform



from tqdm import tqdm

from glob import glob





from keras.models import Model, Sequential

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Masking

from keras.utils import np_utils, to_categorical







from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

#copying the pretrained models to the cache directory

cache_dir = os.path.expanduser(os.path.join('~', '.keras'))

if not os.path.exists(cache_dir):

    os.makedirs(cache_dir)

models_dir = os.path.join(cache_dir, 'models')

if not os.path.exists(models_dir):

    os.makedirs(models_dir)



#copy the Xception models

!cp ../input/keras-pretrained-models/xception* ~/.keras/models/

#show

!ls ~/.keras/models
dir_kaggle ='../input/face-mask-detection'

data_kaggle ='../input/face-mask-detection/dataset'

with_mask ='..../input/face-mask-detection/dataset/with_mask'

without_mask='../input/face-mask-detection/dataset/without_mask'





class_data= ['with_mask','without_mask']

len_class_data = len(class_data)
image_count = {}

train_data = []



for i , class_data in tqdm(enumerate(class_data)):

    class_folder = os.path.join(data_kaggle,class_data)

    label = class_data

    image_count[class_data] = []

    

    for path in os.listdir(os.path.join(class_folder)):

        image_count[class_data].append(class_data)

        train_data.append(['{}/{}'.format(class_data, path), i, class_data])
#show image count

for key, value in image_count.items():

    print('{0} -> {1}'.format(key, len(value)))
#create a dataframe

df = pd.DataFrame(train_data, columns=['file', 'id', 'label'])

df.shape

df.head()


#masking function

def create_mask_for_image(image):

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)



    lower_hsv = np.array([0,0,250])

    upper_hsv = np.array([250,255,255])

    

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask



#image  deskew function

def  deskew_image(image):

    mask = create_mask_for_image(image)

    output = cv2.bitwise_and(image, image, mask = mask)

    return output/255



#image  gray  function

def  gray_image(image):

    mask = create_mask_for_image(image)

    output = cv2.bitwise_and(image, image, mask = mask)

    return output/255



#image  thresh  function

def  thresh_image(image):

    mask = create_mask_for_image(image)

    output = cv2.bitwise_and(image, image, mask = mask)

    return output/255





#image  rnoise  function

def  rnoise_image(image):

    mask = create_mask_for_image(image)

    output = cv2.bitwise_and(image, image, mask = mask)

    return output/255



#image  dilate  function

def  dilate_image(image):

    mask = create_mask_for_image(image)

    output = cv2.bitwise_and(image, image, mask = mask)

    return output/255





#image  erode  function

def  erode_image(image):

    mask = create_mask_for_image(image)

    output = cv2.bitwise_and(image, image, mask = mask)

    return output/255







#image  opening  function

def  opening_image(image):

    mask = create_mask_for_image(image)

    output = cv2.bitwise_and(image, image, mask = mask)

    return output/255



#image canny function

def  canny_image(image):

    mask = create_mask_for_image(image)

    output = cv2.bitwise_and(image, image, mask = mask)

    return output/255





#image segmentation function

def segment_image(image):

    mask = create_mask_for_image(image)

    output = cv2.bitwise_and(image, image, mask = mask)

    return output/255





#sharpen the image

def sharpen_image(image):

    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)

    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)

    return image_sharp







# function to get an image

def read_img(filepath, size):

    img = image.load_img(os.path.join(data_kaggle, filepath), target_size=size)

    #convert image to array

    img = image.img_to_array(img)

    return img
nb_rows = 3

nb_cols = 5

fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(10, 5));

plt.suptitle('SAMPLE IMAGES');

for i in range(0, nb_rows):

    for j in range(0, nb_cols):

        axs[i, j].xaxis.set_ticklabels([]);

        axs[i, j].yaxis.set_ticklabels([]);

        axs[i, j].imshow((read_img(df['file'][np.random.randint(400)], (255,255)))/255.);

plt.show();
#get an image

img = read_img(df['file'][12],(255,255))



#mask

image_mask = create_mask_for_image(img)



fig, ax = plt.subplots(1, 2, figsize=(5, 5));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('MASK', fontsize=12)



ax[0].imshow(img/255);

ax[1].imshow(image_mask);

#get an image

img = read_img(df['file'][13],(255,255))



#segmentation

image_segmented = segment_image(img)



fig, ax = plt.subplots(1, 2, figsize=(5, 5));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('SEGMENTED', fontsize=12)



ax[0].imshow(img/255);

ax[1].imshow(image_segmented);
#get an image

img = read_img(df['file'][14],(255,255))



#deskew

image_deskew = deskew_image(img)



fig, ax = plt.subplots(1, 2, figsize=(5, 5));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('DESKEW', fontsize=12)



ax[0].imshow(img/255);

ax[1].imshow(image_deskew);
#get an image

img = read_img(df['file'][105],(255,255))



#gray

image_gray = gray_image(img)



fig, ax = plt.subplots(1, 2, figsize=(5, 5));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('GRAY', fontsize=12)



ax[0].imshow(img/255);

ax[1].imshow(image_gray);
#get an image

img = read_img(df['file'][250],(255,255))



#thresh

image_thresh = thresh_image(img)



fig, ax = plt.subplots(1, 2, figsize=(5, 5));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('THRESH', fontsize=12)



ax[0].imshow(img/255);

ax[1].imshow(image_thresh);
#get an image

img = read_img(df['file'][275],(255,255))



#rnoise

image_rnoise = rnoise_image(img)



fig, ax = plt.subplots(1, 2, figsize=(5, 5));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('RNOISE', fontsize=12)



ax[0].imshow(img/255);

ax[1].imshow(image_rnoise);
#get an image

img = read_img(df['file'][15],(255,255))



#canny

image_canny = canny_image(img)



fig, ax = plt.subplots(1, 2, figsize=(5, 5));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('CANNY', fontsize=12)



ax[0].imshow(img/255);

ax[1].imshow(image_canny);
#get an image

img = read_img(df['file'][12],(255,255))



#sharpen the image

image_sharpen = sharpen_image(img)



fig, ax = plt.subplots(1, 2, figsize=(5, 5));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('SHARPEN', fontsize=12)



ax[0].imshow(img/255);

ax[1].imshow(image_sharpen);
#get an image

img = read_img(df['file'][60],(255,255))



#mask

image_mask = create_mask_for_image(img)



#segmentation

image_segmented = segment_image(img)





#deskew

image_deskew = deskew_image(img)



#gray

image_gray = gray_image(img)



#thresh

image_thresh = thresh_image(img)



#rnoise

image_rnoise = rnoise_image(img)



#canny

image_canny = canny_image(img)



#sharpen the image

image_sharpen = sharpen_image(img)



fig, ax = plt.subplots(1, 9, figsize=(15, 6));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=9)

ax[1].set_title('MASK', fontsize=9)

ax[2].set_title('SEGMENTED', fontsize=9)

ax[3].set_title('DESKEW', fontsize=9)

ax[4].set_title('GRAY', fontsize=9)

ax[5].set_title('THREST', fontsize=9)

ax[6].set_title('RNOISE', fontsize=9)

ax[7].set_title('CANNY', fontsize=9)

ax[8].set_title('SHARPEN', fontsize=9)



ax[0].imshow(img/255);

ax[1].imshow(image_mask);

ax[2].imshow(image_segmented);

ax[3].imshow(image_deskew);

ax[4].imshow(image_gray );

ax[5].imshow(image_thresh);

ax[6].imshow(image_rnoise);

ax[7].imshow(image_canny);

ax[8].imshow(image_sharpen);

#get an image

img = read_img(df['file'][321],(255,255))



#erode

image_erode = erode_image(img)



fig, ax = plt.subplots(1, 2, figsize=(5, 5));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('ERODE', fontsize=12)



ax[0].imshow(img/255);

ax[1].imshow(image_erode);
#get an image

img = read_img(df['file'][175],(255,255))



#dilate

image_dilate = dilate_image(img)



fig, ax = plt.subplots(1, 2, figsize=(5, 5));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('DILATE', fontsize=12)



ax[0].imshow(img/255);

ax[1].imshow(image_dilate);
#get an image

img = read_img(df['file'][55],(255,255))



#opening

image_opening = opening_image(img)



fig, ax = plt.subplots(1, 2, figsize=(5, 5));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('OPENING', fontsize=12)



ax[0].imshow(img/255);

ax[1].imshow(image_opening);
#get an image

img = read_img(df['file'][60],(255,255))



#dilate

image_dilate = dilate_image(img)



#erode

image_erode = erode_image(img)



#opening

image_opening = opening_image(img)



fig, ax = plt.subplots(1, 4, figsize=(10, 6));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=9)

ax[1].set_title('DILATE', fontsize=9)

ax[2].set_title('ERODE', fontsize=9)

ax[3].set_title('OPENING', fontsize=9)





ax[0].imshow(img/255);

ax[1].imshow(image_dilate);

ax[2].imshow(image_erode);

ax[3].imshow(image_opening);



#get an image

img = read_img(df['file'][135],(255,255))



#Blur

blur = cv2.blur(img,(5,5))



fig, ax = plt.subplots(1, 2, figsize=(5, 5));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('BLUR IMAGE', fontsize=12)



ax[0].imshow(img/255);

ax[1].imshow(blur);
#get an image

img = read_img(df['file'][5],(255,255))



#GaussianBlur

Gblur = cv2.GaussianBlur(img,(5,5),0)



fig, ax = plt.subplots(1, 2, figsize=(5, 5));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('GAUSSIAN BLUR', fontsize=12)



ax[0].imshow(img/255);

ax[1].imshow(Gblur);
#get an image

img = read_img(df['file'][10],(255,255))



#medianBlur

blur_image_median = cv2.medianBlur(img,5)



fig, ax = plt.subplots(1, 2, figsize=(5, 5));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('MEDIAN BLUR', fontsize=12)



ax[0].imshow(img/255);

ax[1].imshow(blur_image_median);


#get an image

img = read_img(df['file'][15],(255,255))



#BILATERAL FILTER

bilblur = cv2.bilateralFilter(img,9,75,75)



fig, ax = plt.subplots(1, 2, figsize=(5, 5));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=12)

ax[1].set_title('BILATERAL FILTER', fontsize=12)



ax[0].imshow(img/255);

ax[1].imshow(bilblur);

#get an image

img = read_img(df['file'][20],(255,255))



fig, ax = plt.subplots(1, 5, figsize=(10, 6));

plt.suptitle('RESULT', x=0.5, y=0.8)

plt.tight_layout(1)



ax[0].set_title('ORIGINAL', fontsize=9)

ax[1].set_title('BLUR IMAGE', fontsize=9)

ax[2].set_title('GAUSSIAN BLUR', fontsize=9)

ax[3].set_title('MEDIAN BLUR', fontsize=9)

ax[4].set_title('BILATERAL FILTER', fontsize=9)





ax[0].imshow(img/255);

ax[1].imshow(blur);

ax[2].imshow(Gblur);

ax[3].imshow(blur_image_median);

ax[4].imshow(bilblur);
