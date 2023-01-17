# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#   for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

from matplotlib import pyplot as plt

import cv2
SEED=42
df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')



x = df_train['id_code']

y = df_train['diagnosis']



x, y = shuffle(x, y, random_state=SEED)
print("Total images:",len(df_train)+len(df_test))

print("Total Training images:",len(x))

print("Total testing images:",len(df_test))
df_train.head()
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.20,

                                                      stratify=y, random_state=SEED)

print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

train_y.hist()

valid_y.hist()
%%time

IMG_SIZE = 256

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
#removing noise using gaussian blur - gray version

fig = plt.figure(figsize=(25, 16))

for class_id in sorted(train_y.unique()):

    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(5, random_state=SEED).iterrows()):

        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , IMG_SIZE/10) ,-4 ,128) # the trick is to add this line

        a = image.shape

        plt.imshow(image, cmap='gray')

        ax.set_title('Label: %d-%d-%s' % (class_id, idx, row['id_code']) )

print(a)
#removing noise using gaussian blur - color version

def load_ben_color(path, sigmaX=10):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

        

    return image
%%time



fig = plt.figure(figsize=(25, 16))

for class_id in sorted(train_y.unique()):

    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(5, random_state=SEED).iterrows()):

        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        image = load_ben_color(path,sigmaX=30)

        a = image.shape

        plt.imshow(image)

        ax.set_title('%d-%d-%s' % (class_id, idx, row['id_code']) )

print(a)
#removing noise using gaussian blur - color version with removing uninormative black part



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
def load_ben_color(path, sigmaX=10):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

        

    return image
%%time



fig = plt.figure(figsize=(25, 16))

for class_id in sorted(train_y.unique()):

    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(5, random_state=SEED).iterrows()):

        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        image = load_ben_color(path,sigmaX=30)

        a = image.shape

        plt.imshow(image)

        ax.set_title('%d-%d-%s' % (class_id, idx, row['id_code']) )

print(a)
#circular crop

def circle_crop(img, sigmaX=10):     

    

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

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)

    return img 
%%time



fig = plt.figure(figsize=(25, 16))

for class_id in sorted(train_y.unique()):

    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(5, random_state=SEED).iterrows()):

        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        image = circle_crop(path,sigmaX=30)

        a = image.shape

        plt.imshow(image)

        ax.set_title('%d-%d-%s' % (class_id, idx, row['id_code']) )

print(a)