import math

import os

import cv2

from PIL import Image

import numpy as np

from keras import layers

from keras.applications import DenseNet121

from keras.applications import DenseNet201

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score

import scipy

from tqdm import tqdm

import time

from keras.models import load_model

from keras.preprocessing import image 
model = load_model('../input/dr-detection/dr_model.h5')

model.summary()
from keras.models import Model  

dr_model = Model(inputs=model.layers[0].layers[0].input, 

                 outputs=(model.layers[0].layers[-1].output)) 
def GetCAM(model,cmodel,img_path):

    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)

    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor

    x = np.expand_dims(x, axis=0)

    pred_vec = model.predict(x)

    pred=np.argmax(pred_vec)

    print(img_path)

    if pred==1:

        print("==> DR")

    else:

        print("==> NODR")

    last_conv_output = dr_model.predict(x)

    last_conv_output = np.squeeze(last_conv_output)

    # bilinear upsampling to resize each filtered image to size of original image 

    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1) 

    # dim: 224 x 224 x 1048

    #print(mat_for_mult.shape)

    # get fc layer weights

    all_amp_layer_weights = model.layers[-1].get_weights()[0]

    amp_layer_weights = all_amp_layer_weights[:, pred]

    # get class activation map for object class that is predicted to be in the image

    cam = np.dot(mat_for_mult.reshape((224*224, 1024)), amp_layer_weights).reshape(224,224) 

    # dim: 224 x 224

    return cam

    

    
cam=GetCAM(model,dr_model,"../input/aptos2019-blindness-detection/train_images/0083ee8054ee.png")
plt.imshow(cam)
im = cv2.resize(cv2.cvtColor(cv2.imread("../input/aptos2019-blindness-detection/train_images/0097f532ac9f.png"), cv2.COLOR_BGR2RGB), (224, 224))
fig, ax = plt.subplots()

ax.imshow(im, alpha=1)

ax.imshow(cam, cmap='jet', alpha=0.1)
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
fig = plt.figure(figsize=(32, 32))

num_samples=2



for class_id in sorted(train_df['diagnosis'].unique()):

    for i, (idx, row) in enumerate(train_df.loc[train_df['diagnosis'] == class_id].sample(num_samples).iterrows()):

        ax = fig.add_subplot(5, num_samples, class_id *num_samples + i + 1, xticks=[], yticks=[])

        im = cv2.resize(cv2.cvtColor(cv2.imread(f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"), cv2.COLOR_BGR2RGB), (224, 224))

        cam=GetCAM(model,dr_model,f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png")

        #plt.imshow(im)

        ax.imshow(im, alpha=1)

        ax.imshow(cam, cmap='jet', alpha=0.3)

        ax.set_title(f'Label: {class_id}')