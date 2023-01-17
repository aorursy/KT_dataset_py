import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Input, Flatten, Dense, Dropout

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input



import random



#load the model

model = load_model('../input/owl-vs-frogmoutn-vgg16-model/owl_vs_frogmouth_vgg16.h5')

model.summary()
#now create model where the input would be from the loaded model and the output would be all the conv layers

convlayers = []

layernames = []

for layer in model.layers:

    if ('conv' in layer.name):

        layernames.append(layer.name)

        convlayers.append(layer.output)

        



len(convlayers)



visual_model = Model(inputs=model.input, outputs=convlayers)



visual_model.summary()
#now grab a picture of some owl

!wget 'https://img.apmcdn.org/dd43b8ed9eb761d047ea4f2af7bc3f4985b5d5a5/normal/393f07-20170310-owl-baiting01.jpg'

#resample it

img = image.load_img('393f07-20170310-owl-baiting01.jpg', target_size=(150,150))

image_tensor = image.img_to_array(img)

image_tensor = np.expand_dims(image_tensor, axis=0)

image_tensor /= 255.



image_tensor.shape
#plot the image

plt.imshow(image_tensor[0])
#now feed it into the visualization model

activations = visual_model.predict(image_tensor)

len(activations)
#let's visualize 4 random activation channel of each layer



for layer, name in zip(activations, layernames):

    channels = layer.shape[-1]

    size = layer.shape[1]

    merged_image = np.zeros((size, size*4))

    plt.title(name)

    for image_count in range(4):

        rand_channel = random.randint(1, channels-1)

        plt.imshow(layer[0,:,:,rand_channel], cmap='viridis')

        plt.show()

        

    
