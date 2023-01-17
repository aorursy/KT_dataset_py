# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/sample-images'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow import keras

import cv2

import json

import imageio

import glob



from IPython.display import Image

import matplotlib.pyplot as plt

import matplotlib.cm as cm



FILE_PATH = "/kaggle/input/sample-images/"
display(Image(FILE_PATH+"vgg16_dock.png"))
display(Image('http://gradcam.cloudcv.org/static/images/network.png'))
display(Image("https://www.researchgate.net/profile/Max_Ferguson/publication/322512435/figure/fig3/AS:697390994567179@1543282378794/Fig-A1-The-standard-VGG-16-network-architecture-as-proposed-in-32-Note-that-only.png"))
display(Image(FILE_PATH+"xception.png"))
properties = {

    "vgg16": {

        "img_size": (224, 224),

        "last_conv_layer": "block5_conv3",

        "last_classifier_layers": [

            "block5_pool",

            "flatten",

            "fc1",

            "fc2",

            "predictions",

        ],

        "model_builder": keras.applications.vgg16.VGG16,

        "preprocess_input": keras.applications.vgg16.preprocess_input,

        "decode_predictions": keras.applications.vgg16.decode_predictions,

    },

    "xception": {

        "img_size": (299, 299),

        "last_conv_layer": "block14_sepconv2_act",

        "last_classifier_layers": [

            "avg_pool",

            "predictions",

        ],

        "model_builder": keras.applications.xception.Xception,

        "preprocess_input": keras.applications.xception.preprocess_input,

        "decode_predictions": keras.applications.xception.decode_predictions,

        

    }

}
NETWORK = "vgg16"

IMG_PATH = FILE_PATH + "pier.jpg"

IMG_2_PATH = FILE_PATH + "dock.jpg"

IMG_SIZE = properties[NETWORK]["img_size"]

LAST_CONV_LAYER = properties[NETWORK]["last_conv_layer"]

CLASSIFIER_LAYER_NAMES = properties[NETWORK]["last_classifier_layers"]



TOP_N = 8
model_builder = properties[NETWORK]["model_builder"]

preprocess_input = properties[NETWORK]["preprocess_input"]

decode_predictions = properties[NETWORK]["decode_predictions"]
display(Image(IMG_PATH))
def get_img_array(img_path, size):

    img = keras.preprocessing.image.load_img(img_path, target_size=size)

    array = keras.preprocessing.image.img_to_array(img)

    array = np.expand_dims(array, axis=0)

    return array



def load_imagenet_classes(filepath=FILE_PATH + "imagenet_1000_idx.js"):

    

    with open(filepath, 'r') as file:

        class_dict = json.loads(file.read())

    dict_by_name = {class_dict[key].split(",")[0]: int(key) for key in class_dict}

    return dict_by_name, class_dict



DICT_BY_NAME, CLASS_DICT = load_imagenet_classes()
def get_predictions(image_path, image_size, top_n):

    img_array = get_img_array(image_path, size=image_size)

    img_array = preprocess_input(img_array)

    model = model_builder(weights="imagenet")

    preds = model.predict(img_array)

    preds_n = decode_predictions(preds, top=top_n)[0]

    return preds_n



preds_n = get_predictions(IMG_PATH, IMG_SIZE, TOP_N)
def print_predictions(predictions):

    print("Predictions")

    for index in np.arange(len(predictions)):

        print(f'Id: {DICT_BY_NAME[predictions[index][1]]} Probability: {predictions[index][2]:4f} Class Name: {predictions[index][1].capitalize()}')

print_predictions(preds_n)
def get_top_predicted_indices(predictions, top_n):

    return np.argsort(-predictions).squeeze()[:top_n]



def make_gradcam_heatmap(

    img_array, model, 

    last_conv_layer_name, 

    classifier_layer_names,

    top_n,

    class_indices

):

    #1. Create a model that maps the input image to the activations of the last convolution layer - Get last conv layer's output dimensions

    last_conv_layer = model.get_layer(last_conv_layer_name)

    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    

    #2. Create another model, that maps from last convolution layer to the final class predictions - This is the classifier model that calculated the gradient

    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])

    x = classifier_input

    for layer_name in classifier_layer_names:

        x = model.get_layer(layer_name)(x)

    classifier_model = keras.Model(classifier_input, x)

    

    #3. If top N predictions are to be interospected, Get their Imagenet indices else assign the indices given

    if(top_n > 0):

        last_conv_layer_output = last_conv_layer_model(img_array)

        preds = classifier_model(last_conv_layer_output)

        class_indices = get_top_predicted_indices(preds, top_n)

    else:

        top_n = len(class_indices)

    

    #4. Create an array to store the heatmaps

    heatmaps = []

    #5. Iteratively calculate heatmaps for all classes of interest using GradientTape

    for index in np.arange(top_n):

    

        #6. Watch the last convolution output during the prediction process to calculate the gradients

        #7. Compute the activations of last conv layer and make the tape to watch

        with tf.GradientTape() as tape:

            # Compute activations of the last conv layer and make the tape watch it

            last_conv_layer_output = last_conv_layer_model(img_array)

            tape.watch(last_conv_layer_output)



            #8. Get the class predictions and the class channel using the class index

            preds = classifier_model(last_conv_layer_output)

            class_channel = preds[:, class_indices[index]]

            

        #9. Using tape, Get the gradient for the predicted class wrt the output feature map of last conv layer    

        grads = tape.gradient(

            class_channel,

            last_conv_layer_output

        )

        

        #10. Calculate the mean intensity of the gradient over its feature map channel

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))    

        last_conv_layer_output = last_conv_layer_output.numpy()[0]

        pooled_grads = pooled_grads.numpy()

        

        #11. Multiply each channel in feature map array by weight importance of the channel

        for i in range(pooled_grads.shape[-1]):

            last_conv_layer_output[:, :, i] *= pooled_grads[i]



        #12. The channel-wise mean of the resulting feature map is our heatmap of class activation

        heatmap = np.mean(last_conv_layer_output, axis=-1)



        #13. Normalize the heatmap between [0, 1] for ease of visualization

        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)



        heatmaps.append({

            "class_id": class_indices[index],

            "heatmap": heatmap

        })



    return heatmaps
# Calculate Heatmaps for TOP_N Predictions

heatmaps = make_gradcam_heatmap(

    get_img_array(IMG_PATH, IMG_SIZE), 

    model_builder(weights="imagenet"), 

    LAST_CONV_LAYER, 

    CLASSIFIER_LAYER_NAMES, 

    TOP_N, 

    None

)
def superimpose_heatmap(image_path, heatmap):

    img = keras.preprocessing.image.load_img(image_path)

    img = keras.preprocessing.image.img_to_array(img)

    

    # We rescale heatmap to a range 0-255

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap = keras.preprocessing.image.array_to_img(heatmap)

    heatmap = heatmap.resize((img.shape[1], img.shape[0]))

    

    heatmap = keras.preprocessing.image.img_to_array(heatmap)

    superimposed_img = cv2.addWeighted(heatmap, 0.4, img, 0.6, 0)

    superimposed_img = np.uint8(superimposed_img)

    

    return superimposed_img


def display_superimposed_heatmaps(heatmaps, image_path, image_id):

    n = len(heatmaps)

    n_rows = (n // 3) + 1 if n % 3 > 0 else n // 3

    plt.rcParams['axes.grid'] = False

    plt.rcParams['xtick.labelsize'] = False

    plt.rcParams['ytick.labelsize'] = False

    plt.rcParams['xtick.top'] = False

    plt.rcParams['xtick.bottom'] = False

    plt.rcParams['ytick.left'] = False

    plt.rcParams['ytick.right'] = False

    plt.rcParams['figure.figsize'] = [30, 15]

    for index in np.arange(n):

        heatmap = heatmaps[index]["heatmap"]

        class_id = heatmaps[index]["class_id"]

        class_name = CLASS_DICT[str(class_id)].split(",")[0].capitalize()

        superimposed_image = superimpose_heatmap(image_path, heatmap)

        plt.subplot(n_rows, 3, index+1)

        plt.title(f"{class_id}, {class_name}", fontsize= 30)

        plt.imshow(superimposed_image)

        

    plt.show()

display_superimposed_heatmaps(heatmaps, IMG_PATH, 1)


display(Image(IMG_2_PATH))
#Classes of Interest

class_names = ('dock', 'pier', 'suspension_bridge', 'gondola', 'breakwater', 'dam')

class_indices = np.array([536, 718, 839, 576, 460, 525])



classes = [(class_indices[index], value) for index, value in enumerate(class_names)]

classes
# Calculate Heatmaps for TOP_N Predictions

heatmaps = make_gradcam_heatmap(

    get_img_array(IMG_2_PATH, IMG_SIZE), 

    model_builder(weights="imagenet"), 

    LAST_CONV_LAYER, 

    CLASSIFIER_LAYER_NAMES, 

    0, 

    class_indices

)

display_superimposed_heatmaps(heatmaps, IMG_2_PATH, 2)