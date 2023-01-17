import os

import sys

from keras.layers import Dense, Dropout, GlobalAveragePooling2D, LeakyReLU

from keras.models import Model, Sequential



sys.path.append(os.path.abspath('../input/efficientnet/efficientnet-master/efficientnet-master/'))

from efficientnet import EfficientNetB5



def create_effnetB5_model(input_shape, n_out):

    model = Sequential()

    base_model = EfficientNetB5(weights = None, 

                                include_top = False,

                                input_shape = input_shape)

    base_model.name = 'base_model'

    model.add(base_model)

    model.add(Dropout(0.25))

    model.add(Dense(1024))

    model.add(LeakyReLU())

    model.add(GlobalAveragePooling2D())

    model.add(Dropout(0.5))   

    model.add(Dense(n_out, activation = 'sigmoid'))

    return model
PRETRAINED_MODEL = '../input/efficientnetb5-blindness-detector/blindness_detector_best_qwk.h5'

IMAGE_HEIGHT = 340

IMAGE_WIDTH = 340

num_classes = 5

class_text = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']



print('Creating model...')

model = create_effnetB5_model(input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3), n_out = num_classes)

print('Restoring model from ' + PRETRAINED_MODEL + '...')

model.load_weights(PRETRAINED_MODEL)

model.summary()
def crop_image_from_gray(img, tol = 7):

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

            img = np.stack([img1,img2,img3],axis = -1)

    #         print(img.shape)

        return img

    

def process_image(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)

    return image
import numpy as np

import pandas as pd

import cv2

from tqdm import tqdm_notebook as tqdm



submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

predicted = []



print("Making predictions...")

for i, name in tqdm(enumerate(submit['id_code'])):

    path = os.path.join('../input/aptos2019-blindness-detection/test_images/', name + '.png')

    image = cv2.imread(path)

#    image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    image = process_image(image)

    X = np.array((image[np.newaxis]) / 255)

    raw_prediction = model.predict(X) > 0.5

    prediction = raw_prediction.astype(int).sum(axis = 1) - 1

    predicted.append(prediction[0])
submit['diagnosis'] = predicted

submit.to_csv('submission.csv', index = False)

submit.head(10)