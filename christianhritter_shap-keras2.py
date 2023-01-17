! pip install tensorflow==1.15
import tensorflow as tf

print(tf.version)
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

import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input, decode_predictions

import numpy as np

import shap

import keras.backend as K

import json
# load pre-trained model and choose two images to explain

# VGG16 model, with weights pre-trained on ImageNet.

model = VGG16(weights='imagenet', include_top=True)
model.summary()
# load the ImageNet class names

url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

fname = shap.datasets.cache(url)

with open(fname) as f:

    class_names = json.load(f)

class_names
len(class_names)
X,y = shap.datasets.imagenet50()

X.shape, y.shape
to_explain = X[[39,41]]
y[[39,41]]
plt.imshow(X[39][:,:,0])
plt.imshow(X[41][:,:,0])
idx_class = np.argmax(model.predict(to_explain)[0])

class_names[str(idx_class)]
idx_class = np.argmax(model.predict(to_explain)[1])

class_names[str(idx_class)]
# explain how the input to the 7th layer of the model explains the top two classes

def map2layer(x, layer):

    feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))

    return K.get_session().run(model.layers[layer].input, feed_dict)
e = shap.GradientExplainer((model.layers[7].input, model.layers[-1].output), map2layer(preprocess_input(X.copy()), 7))
shap_values,indexes = e.shap_values(map2layer(to_explain, 7), ranked_outputs=2)
# get the names for the classes

index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)
# plot the explanations

shap.image_plot(shap_values, to_explain, index_names)