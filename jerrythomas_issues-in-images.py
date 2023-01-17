import os

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import json
DSB_DATA_PATH = '../input/data-science-bowl-2018/'

EDA_DATA_PATH = '../input/exploratory-analysis/'

OUTPUT_PATH = './'



TRAIN_PREFIX = 'stage1_train/'

MASK_PATH_EXPR = os.path.join(DSB_DATA_PATH,TRAIN_PREFIX)+'{}/masks/{}.png'

IMAGE_PATH_EXPR = os.path.join(DSB_DATA_PATH,TRAIN_PREFIX)+'{}/images/{}.png'
def show_mask_and_image(image_id,mask_id,cx,cy,radius,enhance=False):

    mask = cv2.imread(MASK_PATH_EXPR.format(image_id,mask_id),cv2.IMREAD_GRAYSCALE)

    image = cv2.imread(IMAGE_PATH_EXPR.format(image_id,image_id))

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))

    if enhance:

        for channel in range(image.shape[2]):

            image[:,:,channel] = clahe.apply(image[:,:,channel])



    fig,ax = plt.subplots(1,2,figsize=(20,20))

    ax[0].imshow(image[cy-radius:cy+radius,cx-radius:cx+radius,:])

    ax[0].set_title('image')

    ax[1].imshow(mask[cy-radius:cy+radius,cx-radius:cx+radius])

    ax[1].set_title('mask')

    plt.show()
image_id = '5d21acedb3015c1208b31778561f8b1079cca7487399300390c3947f691e3974'

mask_id='5e6e650a28e22f651817b2edeacbf93a960adf633f1dbef69ecea585ef35d544'

cx = 385

cy = 490

radius = 55

show_mask_and_image(image_id,mask_id,cx,cy,radius,True)
image_id = '55f98f43c152aa0dc8bea513f8ba558cc57494b81ae4ee816977816e79629c50'

mask_id='e5b2747c30db016c8318c1df1391708b85c290d8a80d62e013b14ceb759c998e'

cx = (102 + 77)//2

cy = (255 + 242)//2

radius = 40

show_mask_and_image(image_id,mask_id,cx,cy,radius)
image_id = 'a0afead3b4fe393f6a6159de040ecb2e66f8a89090abf0d0bf5b8e1d38ae667c'

mask_id='52a0da01e7292a55903c626bad32cb224d74013aed8dcee98b2b5c2ff0d8adc0'

cx = (260 + 281)//2

cy = (357 + 352)//2

radius = 40

show_mask_and_image(image_id,mask_id,cx,cy,radius,True)