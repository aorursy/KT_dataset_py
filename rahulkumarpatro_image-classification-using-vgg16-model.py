# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from keras.applications.vgg16 import VGG16
model = VGG16()
model.summary()
from keras.utils.vis_utils import plot_model
model = VGG16()
plot_model(model, to_file='vgg.png')
from keras.preprocessing.image import load_img
image=load_img('/kaggle/input/sportscar/download (1).jpg',target_size=(224,224))
from IPython.display import Image
Image("/kaggle/input/sportscar/download (1).jpg")
from keras.preprocessing.image import img_to_array

image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
from keras.applications.vgg16 import preprocess_input

image = preprocess_input(image)
yhat = model.predict(image)
from keras.applications.vgg16 import decode_predictions

label = decode_predictions(yhat)

label = label[0][0]
# print the classification

print(label)
print('%s (%.2f%%)' % (label[1], label[2]*100))
from keras.preprocessing.image import load_img
image1=load_img('/kaggle/input/bucket1/download.jpg',target_size=(224,224))
from IPython.display import Image
Image("/kaggle/input/bucket1/download.jpg")
from keras.preprocessing.image import img_to_array

image1 = img_to_array(image1)
image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
from keras.applications.vgg16 import preprocess_input

image1 = preprocess_input(image1)
yhat = model.predict(image1)
from keras.applications.vgg16 import decode_predictions

label = decode_predictions(yhat)

label = label[0][0]
# print the classification

print(label)
print('%s (%.2f%%)' % (label[1], label[2]*100))
from keras.preprocessing.image import load_img
image2=load_img('/kaggle/input/piano1/piano.jpg',target_size=(224,224))
from IPython.display import Image
Image("/kaggle/input/piano1/piano.jpg")
from keras.preprocessing.image import img_to_array

image2 = img_to_array(image2)
image2 = image2.reshape((1, image2.shape[0], image2.shape[1], image2.shape[2]))
from keras.applications.vgg16 import preprocess_input

image2 = preprocess_input(image2)
yhat = model.predict(image2)
from keras.applications.vgg16 import decode_predictions

label = decode_predictions(yhat)

label = label[0][0]
# print the classification

print(label)
print('%s (%.2f%%)' % (label[1], label[2]*100))