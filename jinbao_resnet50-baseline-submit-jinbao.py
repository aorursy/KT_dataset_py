# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from PIL import Image

import cv2

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from tqdm import tqdm

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input, Dense, PReLU, Dropout

from keras.models import Model

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

from keras.optimizers import SGD, Adam

from keras.applications.xception import Xception

from keras.applications.resnet50 import ResNet50

from keras.applications.inception_v3 import InceptionV3

from keras.callbacks import Callback, ModelCheckpoint

from keras.layers import GlobalAveragePooling2D

from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report

%matplotlib inline
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
IMG_SIZE = 224

batch_size = 32

epochs = 10
def preprocess_image(img_path):

    image = cv2.imread(img_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 30) ,-4 ,128)

    return image
def getResNet50(input_shape=(224, 224, 3), classes = 5, weights = None):

    input_layer = Input(shape=input_shape)

    resNet50 = ResNet50(include_top=False, weights=weights)(input_layer)

    x = GlobalAveragePooling2D(name='avg_pool')(resNet50)

    x = Dense(1024, name = 'fc1')(x)

    x = PReLU()(x)

    x = Dropout(0.5)(x)

    x = Dense(classes, activation='softmax', name='output')(x)

    model = Model(input_layer, x)

    return model
resNet50 = getResNet50(weights=None)

resNet50.load_weights("../input/aptos2019-resnet50/resNet50.h5")
submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

ans = []

for i, name in tqdm(enumerate(submit['id_code'])):

    img_path = os.path.join('../input/aptos2019-blindness-detection/test_images/', name+'.png')

    img = preprocess_image(img_path)

    img = np.array(img) * 1.0 / 255

    x = np.expand_dims(img, axis=0)

    pre = resNet50.predict(x)

    ans.append(np.argmax(pre))
submit['diagnosis'] = ans

submit.to_csv('submission.csv', index=False)