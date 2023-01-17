import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from keras.models import load_model

from keras.preprocessing import image



model = load_model('../input/novartis-session-1/amazon_forest.h5')
import cv2



df_train = pd.read_csv('../input/planet-understanding-the-amazon-from-space/train_v2.csv')

df_train['deforest'] = df_train['tags'].str.contains("agriculture|habitation|road|cultivation|slash_burn|conventional_mine|bare_ground|artisinal_mine|selective_logging|blow_down")



new_style = {'grid': False}

plt.rc('axes', **new_style)



for f, l, j in df_train.iloc[[35621]].values:

    img = cv2.imread('../input/planet-understanding-the-amazon-from-space/train-jpg/{}.jpg'.format(f))

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation='nearest', aspect='auto')

    plt.title(('{} - {}'.format(l, j)))



plt.tight_layout()
test_image = image.load_img('../input/planet-understanding-the-amazon-from-space/train-jpg/train_35621.jpg', target_size=(150, 150))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis=0)

test_image = test_image.reshape(1, 150, 150, 3)   



result = np.array(model.predict(test_image))

classes = result.item(0)



if classes == 0:

    print ("Great! Forest is still well preserved!") 

elif classes == 1:

    print ("Oh no! Deforestation event suspected! Send some guards there!")