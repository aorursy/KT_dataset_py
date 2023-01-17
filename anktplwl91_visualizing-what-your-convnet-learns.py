# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions

%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
image_dir = "../input/flowers/flowers"
image_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, rescale=1./255, validation_split=0.15)  

train_gen = image_datagen.flow_from_directory(image_dir, target_size=(299, 299), batch_size=32, class_mode="categorical", subset="training")
valid_gen = image_datagen.flow_from_directory(image_dir, target_size=(299, 299), batch_size=32, class_mode="categorical", subset="validation")
inp = Input((299, 299, 3))
inception = InceptionV3(include_top=False, weights='imagenet', input_tensor=inp, input_shape=(299, 299, 3), pooling='avg')
x = inception.output
x = Dense(256, activation='relu')(x)
x = Dropout(0.1)(x)
out = Dense(5, activation='softmax')(x)

complete_model = Model(inp, out)

complete_model.compile(optimizer='adam', loss='categorical_crossentropy')
#complete_model.summary()
# Running the model using the fit_generater method for 10 epochs
history = complete_model.fit_generator(train_gen, steps_per_epoch=115, epochs=10, validation_data=valid_gen, validation_steps=20, verbose=1)
# Taking the outputs of first 100 layers from trained model, leaving the first Input layer, in a list
layer_outputs = [layer.output for layer in complete_model.layers[1:100]]

# This is image of a Rose flower from our dataset. All of the visualizations in this cell are of this image.
test_image = image_dir+"/rose/2258973326_03c0145f15_n.jpg"

# Loading the image and converting it to a numpy array for feeding it to the model. Its important to use expand_dims since our original model takes batches of images
# as input, and here we are feeding a single image to it, so the number of dimensions should match for model input.
img = image.load_img(test_image, target_size=(299, 299))
img_arr = image.img_to_array(img)
img_arr = np.expand_dims(img_arr, axis=0)
img_arr /= 255.

# Defining a new model using original model's input and all the 100 layers outputs and then predicting the values for all those 100 layers for our test image.
activation_model = Model(inputs=complete_model.input, outputs=layer_outputs)
activations = activation_model.predict(img_arr)

# These are names of layers, the outputs of which we are going to visualize.
layer_names = ['conv2d_1', 'activation_1', 'conv2d_4', 'activation_4', 'conv2d_9', 'activation_9']
activ_list = [activations[0], activations[2], activations[10], activations[12], activations[17], activations[19]]
# Visualization of the activation maps from first convolution layer. Different filters activate different parts of the image, like some are detecting edges, some are
# detecting background, while others are detecting just the outer boundary of the flower and so on.
fig = plt.figure(figsize=(22, 3))
for img in range(30):
    ax = fig.add_subplot(2, 15, img+1)
    ax = plt.imshow(activations[0][0, :, :, img], cmap='plasma')
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
# This is the visualization of activation maps from third convolution layer. In this layer the abstraction has increased. Filters are now able to regognise the edges
# of the flower more closely. Some filters are activating the surface texture of the image as well
fig = plt.figure(figsize=(22, 6))
for img in range(60):
    ax = fig.add_subplot(4, 15, img+1)
    ax = plt.imshow(activations[6][0, :, :, img], cmap='plasma')
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
# These are activation maps from fourth convolution layer. The images have become a little blurry, because of the MaxPooling operation done just before this layer. As
# more Pooling layers are introduced the knowledge reaching the convolution layer becomes more and more abstract, which helps the complete network to finally classify
# the image properly, but visually they don't provide us with much information.
fig = plt.figure(figsize=(22, 6))
for img in range(60):
    ax = fig.add_subplot(4, 15, img+1)
    ax = plt.imshow(activations[10][0, :, :, img], cmap='plasma')
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
# These are the activation maps from next convolution layer after next MaPooling layer. The images have become more blurry
fig = plt.figure(figsize=(22, 6))
for img in range(60):
    ax = fig.add_subplot(4, 15, img+1)
    ax = plt.imshow(activations[17][0, :, :, img], cmap='plasma')
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
# Activation maps from first Concatenate layer Mixed0, which concatenates the ReLU activated outputs from four convolution layers.
fig = plt.figure(figsize=(22, 6))
for img in range(60):
    ax = fig.add_subplot(4, 15, img+1)
    ax = plt.imshow(activations[39][0, :, :, img], cmap='plasma')
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
# The purpose of this function is to just convert a numpy array to a standard image format, so that it can be displayed and viewed comfortably
def deprocess_image(x):
    
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')

    return x
# This function is used to create a loss function that maximizes the value of a given filter in a convolution layer, and then we use SGD to adjust the values of the
# input image so as to maximize this activation value. We pass the layer name and the filter index to the function as arguments. 'loss' is the mean for that particular
# filter, 'grads' is the gradient calculated for this loss with respect to input image. Finally, SGD is run for 80 iterations which continuously maximizes the response
# to input image by adding the gradient. Finally, it uses 'deprocess_image' to convert this array to a representable image format.

def generate_pattern(layer_name, filter_index, size=150):
    
    layer_output = complete_model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, complete_model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([complete_model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(80):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)
# Below are the patterns to which the filters from first convolution layer get activated. As we can see these are very basic cross-sectional patterns formed by
# horizontal and vertical lines, which is what the these filters look in the input image and get activated if they find one.
fig = plt.figure(figsize=(15, 12))
for img in range(30):
    ax = fig.add_subplot(5, 6, img+1)
    ax = plt.imshow(generate_pattern('conv2d_1', img))
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
# Here are patterns to which filters from third convolution layer respond to. These patterns are liitle more abstract than the simple cross-sectional patterns we saw
# for first layer. This tells us that this layer is looking for more deeper and complex patterns than the earlier convolutional layer.
fig = plt.figure(figsize=(25, 13))
for img in range(50):
    ax = fig.add_subplot(5, 10, img+1)
    ax = plt.imshow(generate_pattern('conv2d_3', img))
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
fig = plt.figure(figsize=(15, 13))
for img in range(60):
    ax = fig.add_subplot(6, 10, img+1)
    ax = plt.imshow(generate_pattern('conv2d_4', img))
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
fig = plt.figure(figsize=(15, 13))
for img in range(60):
    ax = fig.add_subplot(6, 10, img+1)
    ax = plt.imshow(generate_pattern('conv2d_9', img))
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
img_path = image_dir+'/daisy/9158041313_7a6a102f7a_n.jpg'

img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = complete_model.predict(x)
preds
# 0 is the class index for class 'Daisy' in Flowers dataset
flower_output = complete_model.output[:, 0]
last_conv_layer = complete_model.get_layer('mixed10')

grads = K.gradients(flower_output, last_conv_layer.output)[0]                               # Gradient of output with respect to 'mixed10' layer
pooled_grads = K.mean(grads, axis=(0, 1, 2))                                                # Vector of size (2048,), where each entry is mean intensity of
                                                                                            # gradient over a specific feature-map channel
iterate = K.function([complete_model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

#2048 is the number of filters/channels in 'mixed10' layer
for i in range(2048):                                                                       # Multiplies each channel in feature-map array by "how important this
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]                           # channel is" with regard to the class
        
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)                                                            # Following two lines just normalize heatmap between 0 and 1
heatmap /= np.max(heatmap)

plt.imshow(heatmap)
img = plt.imread(img_path)
extent = 0, 300, 0, 300
fig = plt.Figure(frameon=False)

img1 = plt.imshow(img, extent=extent)
img2 = plt.imshow(heatmap, cmap='viridis', alpha=0.4, extent=extent)

plt.xticks([])
plt.yticks([])
plt.show()
img_path = image_dir+'/dandelion/458011386_ec89115a19.jpg'

img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = complete_model.predict(x)
preds
#1 is the class index for class 'Dandelion' in Flowers dataset
flower_output = complete_model.output[:, 1]
last_conv_layer = complete_model.get_layer('mixed10')

grads = K.gradients(flower_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([complete_model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

#2048 is the number of filters/channels in 'mixed10' layer
for i in range(2048):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

plt.imshow(heatmap)
img = plt.imread(img_path)

plt.subplot(121)
plt.imshow(img)
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(heatmap)
plt.xticks([])
plt.yticks([])

plt.show()
img = plt.imread(img_path)
extent = 0, 300, 0, 300
fig = plt.Figure(frameon=False)

img1 = plt.imshow(img, extent=extent)
img2 = plt.imshow(heatmap, cmap='viridis', alpha=0.4, extent=extent)

plt.xticks([])
plt.yticks([])
plt.show()