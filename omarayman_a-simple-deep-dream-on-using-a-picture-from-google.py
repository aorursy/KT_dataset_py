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
from keras import backend as k
from keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt
import os
from glob import glob

img=plt.imread( "../input/i.jpg")
plt.imshow(img)

def preprocess(img):
    img4d = img.copy()
    img4d = img4d.astype("float64")
    # add a dimension to img  
    img4d = np.expand_dims(img4d,axis=0)

    if k.image_dim_ordering() =="th":
        #th is order of the image array(channels,width,height)
        img4d = img4d.transpose((1,2,0))# if so change it to width,height and channels 
        img4d = np.expand_dims(img4d,axis=0)
        img4d = vgg16.preprocess_input(img4d) # apply preprocessing to the image to be able to be fed to the model
    return img4d
def deprocess(img4d):
    img = img4d.copy()
    if k.image_dim_ordering() =="th":
        img = img.reshape((img4d.shape[1],img4d.shape[2],img4d.shape[3])) # same as process
        img = img.transpose((1,2,0))
    else:
        img = img.reshape((img4d.shape[1], img4d.shape[2],img4d.shape[3]))
    img[:,:,0]+=103.939
    img[:,:,1] +=116.779
    img[:,:,2]+=123.68
    img = img[:,:,::-1]
    img=np.clip(img,0,255).astype("uint8")
    return img

from keras import Input
img_copy = img.copy() # get a copy from the image then call the preprocess function
print("Original image shape:",img.shape)
p_img = preprocess(img_copy) #img_noise in case of the noise image
batch_shape = p_img.shape
dream = Input(batch_shape=batch_shape) #we have to state the input shape first that our network will deal with 
model = InceptionV3(input_tensor=dream,weights="imagenet",include_top=False) #don't forget to check that the internet is connected on your kernel
model.summary()
layer_dict = {layer.name : layer for layer in model.layers} #see the layers name
print(layer_dict)
num_pool_layers = 5
num_iters_per_layer = 3
step=100



for i in range(num_pool_layers):
    layer_name = "conv2d_{:d}".format(i+1)
    layer_output = layer_dict[layer_name].output #state the output of each layer
    loss = k.mean(layer_output[:,:,:,30 ]) #you can play around with this number as it's nth but a certain filter that get activated with a certain feature in the ImageNet data set like 24 is activated by an Image of elephant, che
    #check https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    grads = k.gradients(loss,dream)[0] #calculate gradients with respect to loss and INPUT
    grads /= (k.sqrt(k.mean(k.square(grads)))+1e-5)
    f = k.function([dream],[loss,grads])  # f here behaves as a function that takes input dream and return outputs loss and grads
    img_value = p_img.copy()
    fig,axes = plt.subplots(1,num_iters_per_layer,figsize=(20,10))
    for it in range(num_iters_per_layer):
        loss_value,grads_value = f([img_value])
        img_value+=grads_value*step
        axes[it].imshow(deprocess(img_value))
    plt.show()
img_noise = np.random.randint(100,150,size=(227,227,3),dtype=np.uint8)
plt.imshow(img_noise)

