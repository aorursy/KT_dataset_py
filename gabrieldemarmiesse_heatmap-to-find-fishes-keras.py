import numpy as np

from keras.layers.core import  Lambda, Merge

from keras.layers.convolutional import Convolution2D

from keras import backend as K

from keras.engine import Layer

from os import listdir

from os.path import isfile, join, dirname

from scipy.io import loadmat

import gc

from keras.utils.layer_utils import layer_from_config

from keras.models import Model

from keras.layers import *



# Credits to heuritech for their great code which was a great inspiration.

# Some of the code comes directly from their repository.

# You can look it up: https://github.com/heuritech/convnets-keras



	

# Keras doesn't have a 4D softmax. So we need this.

class Softmax4D(Layer):

    def __init__(self, axis=-1,**kwargs):

        self.axis=axis

        super(Softmax4D, self).__init__(**kwargs)



    def build(self,input_shape):

        pass



    def call(self, x,mask=None):

        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))

        s = K.sum(e, axis=self.axis, keepdims=True)

        return e / s



    def get_output_shape_for(self, input_shape):

        return input_shape

		



def get_dim(model, layer_index, input_shape=None):

    

    # Input shape is the shape of images used during training.

    if input_shape is not None:

        dummy_vector = np.zeros((1,) + input_shape)

    else:

        if model.layers[0].input_shape[2] is None:

            raise ValueError('You must provide \"input_shape = (3,256,256)\" for example when calling the function.')

        dummy_vector = np.zeros((1,) + model.layers[0].input_shape[1:])

    

    intermediate_layer_model = Model(input=model.input,

                                 output=model.layers[layer_index].output)

    

    out = intermediate_layer_model.predict(dummy_vector)

    

    return out.shape[1:]

	



def from_config(layer, config_dic):

    config_correct = {}

    config_correct['class_name'] = type(layer)

    config_correct['config'] = config_dic

    return layer_from_config(config_correct)

	



def add_to_model(x, layer):

    new_layer = from_config(layer, layer.get_config())

    x = new_layer(x)

    if layer.get_weights() is not None:

        new_layer.set_weights(layer.get_weights())

    return x

	



def layer_type(layer):

    return str(layer)[10:].split(" ")[0].split(".")[-1]

	



def detect_configuration(model):

    # must return the configuration and the number of the first pooling layer

    

    # Names (types) of layers from end to beggining

    inverted_list_layers = [layer_type(layer) for layer in model.layers[::-1]]

    

    layer1 = None

    layer2 = None 

    

    i = len(model.layers)

    

    for layer in inverted_list_layers:

        i -= 1

        if layer2 is None:

            if layer == "GlobalAveragePooling2D" or layer == "GlobalMaxPooling2D":

                layer2 = layer



            elif layer == "Flatten":

                return "local pooling - flatten", i-1

            

        else:

            layer1 = layer

            break

            

    if layer1 == "MaxPooling2D" and layer2 == "GlobalMaxPooling2D":

        return "local pooling - global pooling (same type)", i

    elif layer1 == "AveragePooling2D" and layer2 == "GlobalAveragePooling2D":

        return "local pooling - global pooling (same type)", i

    

    elif layer1 == "MaxPooling2D" and layer2 == "GlobalAveragePooling2D":

        return "local pooling - global pooling (different type)", i+1

    elif layer1 == "AveragePooling2D" and layer2 == "GlobalMaxPooling2D":

        return "local pooling - global pooling (different type)", i+1

    

    else:

        return "global pooling", i

		

    

def add_zeros(w, nb_zeros):

    

    n = w.shape[3]

    indexes = np.array(range(1, n))

    w1 = w

    for i in range(nb_zeros):

        w1 = np.insert(w1, indexes + i, 0, axis=2)

    for i in range(nb_zeros):

        w1 = np.insert(w1, indexes + i, 0, axis=3)

    return w1

	

    

def insert_weights(layer, new_layer):

    W,b = layer.get_weights()

    n_filter,previous_filter,ax1,ax2 = new_layer.get_weights()[0].shape

    ax1 = ax2 = int(np.sqrt(layer.get_weights()[0].shape[0]/new_layer.get_weights()[0].shape[1]))

    new_W = W.reshape((previous_filter,ax1,ax2,n_filter))

    new_W = new_W.transpose((3,0,1,2))

    new_W = new_W[:,:,::-1,::-1]

	

    

    if ax1!=1:

        insert_zeros = int((new_layer.get_weights()[0].shape[2] - ax1)/(ax1-1))

        print("insert_zeros=" + str(insert_zeros))

        new_W =  add_zeros(new_W, insert_zeros)

    

    new_layer.set_weights([new_W,b])

	

    

def copy_last_layers(model, begin,x):

    

    i=begin

    

    for layer in model.layers[begin:]:

        if layer_type(layer) == "Dense":

            

            if i == len(model.layers)-1:

                x = add_reshaped_layer(layer,x,1, no_activation=True)

            else:

                x = add_reshaped_layer(layer,x,1)

            

        elif layer_type(layer) == "Dropout":

            pass

                

        elif layer_type(layer) == "Activation" and i == len(model.layers)-1:

            break

               

        else:

            x = add_to_model(x, layer)

        i+=1

    

    x = Softmax4D(axis=1,name="softmax")(x)

    return x

    

                

def add_reshaped_layer(layer, x, size, no_activation=False, add_zeros = None):



    conf = layer.get_config()

    

    if no_activation:

        activation="linear"

    else:

        activation=conf["activation"]

        

    #size = int(np.sqrt(layer.get_weights()[0].shape[0]/conf["output_dim"]))

    

    new_layer = Convolution2D(conf["output_dim"],size,size, activation=activation, name=conf['name'])

         

        

    x= new_layer(x)

    # We transfer the weights:

    insert_weights(layer, new_layer)

    return x

    



def to_heatmap(model, input_shape = None, delete = False):

    

    # there are four configurations possible:

    # global pooling

    # local pooling - flatten

    # local pooling - global pooling (same type)

    # local pooling - global pooling (different type)

    

    model_type, index = detect_configuration(model)

    

    print("Model type detected: " + model_type)

    

    #new_layer.set_weights(model.layers[0].get_weights())

    img_input = Input(shape=(3,None,None))

   

    # Inchanged part:

    middle_model = Model(input=model.layers[1].input, output=model.layers[index-1].output)

    

    x = middle_model(img_input)

    

    print("Model cut at layer: " + str(index))

        

    if model_type == "global pooling":

        x = copy_last_layers(model, index+1,x)

              

    elif model_type == "local pooling - flatten":

        

        layer = model.layers[index]

        dic = layer.get_config()

        add_zeros = dic["strides"][0] - 1

        dic["strides"] = (1,1)

        new_pool = from_config(layer, dic)

        x = new_pool(x)

        

        size = get_dim(model, index, input_shape)[1]

        print("Pool size infered: " + str(size))

        

        conv_size = size + (size-1) * add_zeros

        

        print("New convolution size: " + str(conv_size))

        

        if index+2 != len(model.layers)-1:

            x = add_reshaped_layer(model.layers[index+2],x,conv_size, add_zeros=add_zeros)

        else:

            x = add_reshaped_layer(model.layers[index+2],x,conv_size, add_zeros=add_zeros,no_activation=True)

            

        x = copy_last_layers(model, index+3,x)

        

        

    elif model_type == "local pooling - global pooling (same type)":

        

        

        dim = get_dim(model, index, input_shape=input_shape)



        new_pool_size = model.layers[index].get_config()["pool_size"][0] * dim[1]

        

        print("Pool size infered: " + str(new_pool_size))

        

        x = AveragePooling2D(pool_size=(new_pool_size, new_pool_size), strides=(1,1)) (x)

        x = copy_last_layers(model, index+2,x)

        

        

    elif model_type == "local pooling - global pooling (different type)":

        x= copy_last_layers(model, index+1,x)

    else:

        raise IndexError("no type for model: " + str(model_type))

        

    

    

    if delete:

        del(model)

        gc.collect()

        print("Original model was deleted.")

    

    return Model(img_input, x)
import urllib.request

urllib.request.urlretrieve("https://raw.githubusercontent.com/gabrieldemarmiesse/heatmaps/master/heatmap.py", "heatmap.py")



from heatmap import to_heatmap

from heatmap import synset_to_dfs_ids

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from keras.applications.vgg16 import VGG16

from keras.applications.vgg19 import VGG19

from keras.applications.resnet50 import ResNet50

from keras.applications.inception_v3 import InceptionV3



from keras.preprocessing import image

from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.models import model_from_json
def display_heatmap(new_model, img_path):



    plt.figure()

    img=mpimg.imread(img_path)

    plt.subplot(121)

    plt.imshow(img)

    

    img = image.load_img(img_path)

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)



    out = new_model.predict(x)



    s = "n02512053" # Imagenet code for "fish"

    ids = synset_to_dfs_ids(s)

    heatmap_fish = out[0,ids].sum(axis=0)

    plt.subplot(122)

    plt.imshow(heatmap_fish, interpolation="none")

    plt.show()
model = VGG16()

new_model = to_heatmap(model)
display_heatmap(new_model, "./train/ALB/img_00110.jpg")

display_heatmap(new_model, "./train/ALB/img_00003.jpg")

display_heatmap(new_model, "./train/ALB/img_00085.jpg")
model = ResNet50()

new_model = to_heatmap(model)
display_heatmap(new_model, "./train/ALB/img_00110.jpg")

display_heatmap(new_model, "./train/ALB/img_00003.jpg")

display_heatmap(new_model, "./train/ALB/img_00085.jpg")
# load json and create model

json_file = open('model_2c_10e_R50_1.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

model = model_from_json(loaded_model_json)

# load weights into new model

model.load_weights("model_2c_10e_R50_1.h5")

print("Loaded model from disk")
new_model = to_heatmap(model, input_shape=(3,256,256))
def display_heatmap(new_model, img_path):



    plt.figure()

    img=mpimg.imread(img_path)

    plt.subplot(121)

    plt.imshow(img)

    

    img = image.load_img(img_path)

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)



    out = new_model.predict(x)



    heatmap_fish = out[0,[0]].sum(axis=0)

    plt.subplot(122)

    plt.imshow(heatmap_fish, interpolation="none")

    plt.show()
display_heatmap(new_model, "./train/ALB/img_00110.jpg")

display_heatmap(new_model, "./train/ALB/img_00003.jpg")

display_heatmap(new_model, "./train/ALB/img_00085.jpg")
model = InceptionV3()

new_model = to_heatmap(model)
display_heatmap(new_model, "./train/ALB/img_00110.jpg")

display_heatmap(new_model, "./train/ALB/img_00003.jpg")

display_heatmap(new_model, "./train/ALB/img_00085.jpg")