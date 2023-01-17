# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from keras.preprocessing.image import load_img, img_to_array, array_to_img

import numpy as np

import matplotlib.pyplot as plt

import os

import cv2
def load_imgs(dir):

   val_images, tar_images = [], []

   for filename in os.listdir(dir)[:3000]:

    img = load_img(dir + filename, target_size = (224,224,3))

    pixels = img_to_array(img)

    p1 = (pixels[:,:257] - 127.5) / 127.5

    src_images.append(p1)

    p2 = (pixels[:,257:] - 127.5) / 127.5

    tar_images.append(p2)

   return [np.asarray(val_images), np.asarray(tar_images)]

  
imagepath = "../input/dataset/dataset_updated/training_set/painting/"

image = cv2.imread(imagepath+"1179.jpg")

plt.imshow(image)

img.shape
imagepath=TestImagePath+"15.jpg"

image_for_test = ExtractTestInput(imagepath)

Prediction = Model_Colourization.predict(image_for_test)

Prediction = Prediction*128

Prediction=Prediction.reshape(224,224,2)
plt.figure(figsize=(30,20))

plt.subplot(5,5,1)

img = cv2.imread(TestImagePath+"15.jpg")

img_1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)

img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

img = cv2.resize(img, (224, 224))

plt.imshow(img)

plt.subplot(5,5,1+1)

img_ = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

img_[:,:,1:] = Prediction

img_ = cv2.cvtColor(img_, cv2.COLOR_Lab2RGB)

plt.title("Predicted Image")

plt.imshow(img_)

plt.subplot(5,5,1+2)

plt.title("Ground truth")

plt.imshow(img_1)