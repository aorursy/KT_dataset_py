# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Imports

from keras.layers import Input, Conv2D, MaxPool2D

from keras.layers import Dense, Flatten

from keras.models import Model



# Define the input

#   Unlike the Sequential model, you must create and define 

#   a standalone "Input" layer that specifies the shape of input 

#   data. The input layer takes a "shape" argument, which is a 

#   tuple that indicates the dimensionality of the input data.

#   When input data is one-dimensional, such as the MLP, the shape 

#   must explicitly leave room for the shape of the mini-batch size 

#   used when splitting the data when training the network. Hence, 

#   the shape tuple is always defined with a hanging last dimension.



_input = Input((224,224,1))



conv1 = Conv2D(filters=64, kernel_size=(3,3), padding = "same", activation="relu")(_input)

conv2 = Conv2D(filters=64, kernel_size=(3,3), padding = "same", activation="relu")(conv1)

pool1 = MaxPool2D((2,2))(conv2)





conv3 = Conv2D(filters=128, kernel_size=(3,3), padding = "same", activation="relu")(pool1)

conv4 = Conv2D(filters=128, kernel_size=(3,3), padding = "same", activation="relu")(conv3)

pool2 = MaxPool2D((2,2))(conv4)





conv5 = Conv2D(filters=256, kernel_size=(3,3), padding = "same", activation="relu")(pool2)

conv6 = Conv2D(filters=256, kernel_size=(3,3), padding = "same", activation="relu")(conv5)

conv7 = Conv2D(filters=256, kernel_size=(3,3), padding = "same", activation="relu")(conv6)

pool3 = MaxPool2D((2,2))(conv7)





conv8 = Conv2D(filters=512, kernel_size=(3,3), padding = "same", activation="relu")(pool3)

conv9 = Conv2D(filters=512, kernel_size=(3,3), padding = "same", activation="relu")(conv8)

conv10 = Conv2D(filters=512, kernel_size=(3,3), padding = "same", activation="relu")(conv9)

pool4 = MaxPool2D((2,2))(conv10)





conv11 = Conv2D(filters=512, kernel_size=(3,3), padding = "same", activation="relu")(pool4)

conv12 = Conv2D(filters=512, kernel_size=(3,3), padding = "same", activation="relu")(conv11)

conv13 = Conv2D(filters=512, kernel_size=(3,3), padding = "same", activation="relu")(conv12)

pool5 = MaxPool2D((2,2))(conv13)



flat = Flatten()(pool5)





dense1 = Dense(4096, activation="relu")(flat)

dense2 = Dense(4096, activation="relu")(dense1)

output = Dense(1000, activation="softmax")(dense2)



vgg16_model = Model(inputs = _input, outputs= output)



from keras.applications.vgg16 import decode_predictions, preprocess_input

from keras.preprocessing import image

import matplotlib.pyplot as plt

from PIL import Image

import seaborn as sns





# loadingImages

catImage1 = "../input/dogs-vs-cats-redux-kernels-edition/train/cat.10032.jpg"

dogImage1 = "../input/dogs-vs-cats-redux-kernels-edition/train/dog.10042.jpg"

flowerImage1 = "../input/flowers-recognition/flowers/flowers/rose/16078501836_3ac067e18a.jpg"

fruit1 = "../input/fruits/fruits-360_dataset/fruits-360/Training/Mango/49_100.jpg"



images = [catImage1, dogImage1, flowerImage1, fruit1]



def loadImage(path):

    # changing dimensions to (224,224)     

    img = image.load_img(path, target_size=(224,224))

    # converting image to array

    img = image.img_to_array(img)

    # Expand the shape of an array.

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    return img







def getPredictions(model):

    f,ax = plt.subplots(1,4)

    f.set_size_inches(80, 40)

    for i in range(4):

        ax[i].imshow(Image.open(images[i]).resize((200,200), Image.ANTIALIAS))

    plt.show()

    

    f,axes = plt.subplots(1,4)

    f.set_size_inches(80, 20)

    for i,img_path in enumerate(images):

        img = loadImage(img_path)

        preds = decode_predictions(model.predict(img), top=3)[0]

        b = sns.barplot(y = [c[1] for c in preds], x = [c[2] for c in preds], color="gray", ax= axes[i])

        b.tick_params(labelsize=55)

        f.tight_layout()

from keras.applications.vgg16 import VGG16

vgg16_weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

vgg16_model = VGG16(weights=vgg16_weights)

getPredictions(vgg16_model)
from keras.applications import VGG19

vgg19_weights = '../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5'

vgg19_model = VGG19(weights = vgg19_weights)

getPredictions(vgg19_model)