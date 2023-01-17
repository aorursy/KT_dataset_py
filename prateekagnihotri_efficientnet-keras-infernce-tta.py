# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import glob

glob.glob('../input/prostate-cancer-grade-assessment/*')
!pip install -qq ../input/efficientnet/efficientnet-1.0.0-py3-none-any.whl
import os

import cv2

import skimage.io

import numpy as np

import pandas as pd



from keras.layers import *

from keras.models import Model

import efficientnet.keras as efn

from keras.applications.nasnet import  preprocess_input
I = Input((256,256,3))

efnb3 = efn.EfficientNetB3(weights = None, include_top = False, input_tensor = I, pooling = 'avg', classes = None)

for layer in efnb3.layers:

    layer.trainable = True

x = Dropout(0.5)(efnb3.output)

x = Dense(64, activation='relu')(x)

x = Dense(6,activation='softmax')(x)



model = Model(inputs = efnb3.input, outputs = x)

model.load_weights('../input/model-1/model.h5')
def get_image(img_name, data_dir='../input/prostate-cancer-grade-assessment/test_images'):

    img_path = os.path.join(data_dir, f'{img_name}.tiff')

    img = skimage.io.MultiImage(img_path)

    img = cv2.resize(img[-1], (256,256))

    img = preprocess_input(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    return img
def TTA(img):

    img1 = img

    img2 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    img3 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img4 = cv2.rotate(img, cv2.ROTATE_180)

    images = [img1, img2, img3, img4]

    

    return model.predict(np.array(images), batch_size=4)
def post_process(preds):

    avg = np.sum(preds,axis = 0)

    label = np.argmax(avg)

    return label
from tqdm import tqdm
data_dir = '../input/prostate-cancer-grade-assessment/test_images'

sample_submission = pd.read_csv('../input/prostate-cancer-grade-assessment/sample_submission.csv')

# data_dir = '../input/prostate-cancer-grade-assessment/train_images'

# sample_submission = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv').head(50)



test_images = sample_submission.image_id.values

labels = []



try:    

    for image in tqdm(test_images):

        img = get_image(image, data_dir)

        preds = TTA(img)

        label = post_process(preds)

        labels.append(label)

    sample_submission['isup_grade'] = labels

except:

    print('Test dir not found')

    

sample_submission['isup_grade'] = sample_submission['isup_grade'].astype(int)

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()