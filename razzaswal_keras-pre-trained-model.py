import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import img_to_array, load_img
from IPython.core.display import display
FILE_1 = '../input/keras-pretrained-model/Bhotu.jpg'
img = load_img(FILE_1, target_size=(299,299))
img
img_array = img_to_array(img)
img_array.shape
expanded = np.expand_dims(img_array, axis=0)
expanded.shape
tf.Graph()
from keras.applications.inception_resnet_v2 import InceptionResNetV2
%%time
inception_model = InceptionResNetV2(weights='imagenet')
inception_model.predict(expanded)
from keras.applications.inception_resnet_v2 import decode_predictions
# prediction = inception_model.predict(expanded)
# decode_predictions(prediction)
from keras.applications.inception_resnet_v2 import preprocess_input
preprocessed = preprocess_input(expanded)
preprocessed.shape
prediction = inception_model.predict(preprocessed)
decode_predictions(prediction)
# define a function
def format_img_inceptionresnet(filename):
    img = load_img(filename, target_size=(299,299))
    img_array = img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(expanded)
#to show image first
data = format_img_inceptionresnet('../input/keras-pretrained-model1/bhotu_3mnth.jpg')
prediction = inception_model.predict(data)
display(load_img('../input/keras-pretrained-model1/bhotu_3mnth.jpg'))
decode_predictions(prediction)
data = format_img_inceptionresnet('../input/keras-pretrained-model1/Dromedary-camels.jpg')
prediction = inception_model.predict(data)
display(load_img('../input/keras-pretrained-model1/Dromedary-camels.jpg'))
decode_predictions(prediction)
from keras.applications.vgg16 import VGG16, decode_predictions as decode_vgg, preprocess_input as preprocess_input_vgg
def format_img_vgg16(filename):
    img = load_img(filename, target_size=(224,224,3))
    img_array = img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input_vgg(expanded)
%%time
vgg16_model = VGG16(weights='imagenet')
data = format_img_vgg16('../input/keras-pretrained-model1/africa_cropped.jpg')
prediction = vgg16_model.predict(data)
display(load_img('../input/keras-pretrained-model1/africa_cropped.jpg'))
decode_predictions(prediction)
data = format_img_vgg16('../input/keras-pretrained-model1/322868_1100-1100x628.jpg')
prediction = vgg16_model.predict(data)
display(load_img('../input/keras-pretrained-model1/322868_1100-1100x628.jpg'))
decode_predictions(prediction)
data = format_img_vgg16('../input/keras-pretrained-model1/08 Doorway.jpg')
prediction = vgg16_model.predict(data)
display(load_img('../input/keras-pretrained-model1/08 Doorway.jpg'))
decode_predictions(prediction)
from keras.applications.vgg19 import VGG19, decode_predictions as decode_vgg19, preprocess_input as preprocess_input_vgg19
def format_img_vgg19(filename):
    img = load_img(filename, target_size=(224,224,3))
    img_array = img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input_vgg19(expanded)
%%time
vgg19_model = VGG19(weights='imagenet')
data = format_img_vgg19('../input/keras-pretrained-model1/bhotu_3mnth.jpg')
prediction = vgg19_model.predict(data)
display(load_img('../input/keras-pretrained-model1/bhotu_3mnth.jpg'))
decode_predictions(prediction)
from keras.applications.nasnet import NASNetLarge, decode_predictions as decode_nasnet, preprocess_input as preprocess_input_nasnet
def format_img_nasnet(filename):
    img = load_img(filename, target_size=(331,331,3))
    img_array = img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input_nasnet(expanded)
%%time
nasnet_model = NASNetLarge(weights='imagenet')
data = format_img_nasnet('../input/keras-pretrained-model1/Tower-Bridge.jpg')
prediction = nasnet_model.predict(data)
display(load_img('../input/keras-pretrained-model1/Tower-Bridge.jpg'))
decode_predictions(prediction)
data = format_img_nasnet('../input/keras-pretrained-model1/11 Shoe.jpg')
prediction = nasnet_model.predict(data)
display(load_img('../input/keras-pretrained-model1/11 Shoe.jpg'))
decode_predictions(prediction)
