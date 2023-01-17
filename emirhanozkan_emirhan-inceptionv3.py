import numpy as np

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

import pandas as pd



# fix dimension ordering issue

# https://stackoverflow.com/questions/39547279/loading-weights-in-th-format-when-keras-is-set-to-tf-format

from keras import backend as K

K.set_image_dim_ordering('th')
from keras.applications.inception_v3 import InceptionV3

weights = '../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'

model =InceptionV3(weights=weights)

print (model.summary())
from os import makedirs

from os.path import join, exists, expanduser



cache_dir = expanduser(join('~', '.keras'))

if not exists(cache_dir):

    makedirs(cache_dir)

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)

!cp  ../input/inceptionv3/* ~/.keras/models/
from keras.applications.inception_v3 import preprocess_input, decode_predictions

import time



current_milli_time = lambda: int(round(time.time() * 1000))
from keras.preprocessing import image

img_path = "../input/flowers-recognition/flowers/flowers/dandelion/1074999133_1e4a1e042e.jpg"

img = image.load_img(img_path, target_size=(224, 224))

img
img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)

img = preprocess_input(img)



start = current_milli_time()

preds = model.predict(img)

end = current_milli_time()

print('Predicted in {} ms: {}'.format(end-start, decode_predictions(preds, top=3)[0]))
from keras.preprocessing import image

img_path = "../input/flowers-recognition/flowers/flowers/rose/10503217854_e66a804309.jpg"

img = image.load_img(img_path, target_size=(224, 224))

img
img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)

img = preprocess_input(img)



start = current_milli_time()

preds = model.predict(img)

end = current_milli_time()

print('Predicted in {} ms: {}'.format(end-start, decode_predictions(preds, top=3)[0]))
from keras.preprocessing import image

img_path = "../input/flowers-recognition/flowers/flowers/daisy/34539556222_f7ba32f704_n.jpg"

img = image.load_img(img_path, target_size=(224, 224))

img
img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)

img = preprocess_input(img)



start = current_milli_time()

preds = model.predict(img)

end = current_milli_time()

print('Predicted in {} ms: {}'.format(end-start, decode_predictions(preds, top=3)[0]))
from keras.preprocessing import image

img_path = "../input/flowers-recognition/flowers/flowers/rose/9609569441_eeb8566e94.jpg"

img = image.load_img(img_path, target_size=(224, 224))

img
img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)

img = preprocess_input(img)



start = current_milli_time()

preds = model.predict(img)

end = current_milli_time()

print('Predicted in {} ms: {}'.format(end-start, decode_predictions(preds, top=3)[0]))