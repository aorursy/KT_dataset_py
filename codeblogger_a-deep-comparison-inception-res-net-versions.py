# import the necessary packages

import matplotlib.pyplot as plt 

import tensorflow as tf

from PIL import Image 

import seaborn as sns

import pandas as pd 

import numpy as np

import os 
img1 = "../input/cat-and-dog/training_set/training_set/cats/cat.1915.jpg"

img2 = "../input/cat-and-dog/training_set/training_set/dogs/dog.2881.jpg"

img3 = "../input/flowers-recognition/flowers/flowers/sunflower/7791014076_07a897cb85_n.jpg"

img4 = "../input/fruits/fruits-360/Training/Banana/221_100.jpg"

imgs = [img1, img2, img3, img4]
from tensorflow.keras.applications.inception_v3 import preprocess_input



def _load_image(img_path):

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

    img = tf.keras.preprocessing.image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    return img 



def _get_predictions(_model):

    f, ax = plt.subplots(1, 4)

    f.set_size_inches(80, 40)

    for i in range(4):

        ax[i].imshow(Image.open(imgs[i]).resize((200, 200), Image.ANTIALIAS))

    plt.show()

    

    f, axes = plt.subplots(1, 4)

    f.set_size_inches(80, 10)

    for i,img_path in enumerate(imgs):

        img = _load_image(img_path)

        preds  = tf.keras.applications.imagenet_utils.decode_predictions(_model.predict(img), top=3)[0]

        b = sns.barplot(y=[c[1] for c in preds], x=[c[2] for c in preds], ax=axes[i])

        b.tick_params(labelsize=36)

        f.tight_layout()
from keras.applications.xception import Xception

xception_weights = '../input/xception/xception_weights_tf_dim_ordering_tf_kernels.h5'

xception_model = Xception(weights=xception_weights)

_get_predictions(xception_model)
from tensorflow.keras.applications.vgg16 import preprocess_input



def _load_image(img_path):

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

    img = tf.keras.preprocessing.image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    return img 
from keras.applications.vgg16 import VGG16

vgg16_weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

vgg16_model = VGG16(weights=vgg16_weights)

_get_predictions(vgg16_model)
def _load_image(img_path):

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

    img = tf.keras.preprocessing.image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.vgg19.preprocess_input(img)

    return img
from keras.applications.vgg19 import VGG19

vgg19_weights = '../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5'

vgg19_model = VGG19(weights=vgg19_weights)

_get_predictions(vgg19_model)
def _load_image(img_path):

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

    img = tf.keras.preprocessing.image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.resnet.preprocess_input(img)

    return img 
from keras.applications.resnet50 import ResNet50

resnet_model = ResNet50(weights="imagenet")

_get_predictions(resnet_model)
from tensorflow.keras.applications.xception import preprocess_input



def _load_image(img_path):

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

    img = tf.keras.preprocessing.image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    return img
from keras.applications.xception import Xception

xception_weights = '../input/xception/xception_weights_tf_dim_ordering_tf_kernels.h5'

xception_model = Xception(weights=xception_weights)

_get_predictions(xception_model)
def _load_image(img_path):

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

    img = tf.keras.preprocessing.image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)

    return img
inceptionResNetV2_model = tf.keras.applications.InceptionResNetV2(weights="imagenet")

_get_predictions(inceptionResNetV2_model)
def _load_image(img_path):

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

    img = tf.keras.preprocessing.image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.resnet_v2.preprocess_input(img)

    return img 
resnet50V2_model = tf.keras.applications.ResNet50V2(weights="imagenet", classifier_activation="softmax", input_shape=(224,224,3), pooling='max')

_get_predictions(resnet50V2_model)
def _load_image(img_path):

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

    img = tf.keras.preprocessing.image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.efficientnet.preprocess_input(img)

    return img 
resnet50V2_model = tf.keras.applications.EfficientNetB0(weights="imagenet", classifier_activation="softmax", input_shape=(224,224,3), pooling='max')

_get_predictions(resnet50V2_model)