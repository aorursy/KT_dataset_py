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
import os

import cv2

import random

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, cohen_kappa_score

from sklearn.metrics import confusion_matrix





import keras.backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Input, Dense, Dropout

from keras.applications.inception_v3 import InceptionV3

from keras.applications.inception_v3 import preprocess_input, decode_predictions
train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
train.head()
test.head()
print('Number of train samples: ', train.shape[0])

print('Number of test samples: ', test.shape[0])
sns.set_style("white")

count = 1

plt.figure(figsize=[20, 20])

for img_name in train['id_code'][:15]:

    img = cv2.imread("../input/aptos2019-blindness-detection/train_images/%s.png" % img_name)[...,[2, 1, 0]]

    plt.subplot(5, 5, count)

    plt.imshow(img)

    plt.title("Image %s" % count)

    count += 1

    

plt.show()
train['diagnosis'].value_counts().sort_index().plot(kind="bar", 

                                                       figsize=(12,5), 

                                                       rot=0)

plt.title("Label Distribution (Training Set)", 

          weight='bold', 

          fontsize=18)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel("Label", fontsize=17)

plt.ylabel("Frequency", fontsize=17);
# Preprocecss data

train["id_code"] = train["id_code"].apply(lambda x: x + ".png")

test["id_code"] = test["id_code"].apply(lambda x: x + ".png")

train['diagnosis'] = train['diagnosis'].astype('str')

train.head()
#Model parameters

BATCH_SIZE = 8

EPOCHS = 20

WARMUP_EPOCHS = 2

LEARNING_RATE = 1e-4

WARMUP_LEARNING_RATE = 1e-3

HEIGHT = 512

WIDTH = 512

CANAL = 3

N_CLASSES = train['diagnosis'].nunique()

ES_PATIENCE = 5

RLROP_PATIENCE = 3

DECAY_DROP = 0.5
train_datagen=ImageDataGenerator(rescale=1./255, 

                                 validation_split=0.2,

                                 horizontal_flip=True)



train_generator=train_datagen.flow_from_dataframe(

    dataframe=train,

    directory="../input/aptos2019-blindness-detection/train_images/",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    target_size=(HEIGHT, WIDTH),

    subset='training')



valid_generator=train_datagen.flow_from_dataframe(

    dataframe=train,

    directory="../input/aptos2019-blindness-detection/train_images/",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=BATCH_SIZE,

    class_mode="categorical",    

    target_size=(HEIGHT, WIDTH),

    subset='validation')



test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_dataframe(  

        dataframe=test,

        directory = "../input/aptos2019-blindness-detection/test_images/",

        x_col="id_code",

        target_size=(HEIGHT, WIDTH),

        batch_size=1,

        shuffle=False,

        class_mode=None)
inp = Input((HEIGHT, WIDTH, CANAL))

inception = InceptionV3(include_top=False, weights='imagenet', input_tensor=inp, input_shape=(HEIGHT, WIDTH, CANAL), pooling='avg')

x = inception.output

x = Dense(256, activation='relu')(x)

x = Dropout(0.1)(x)

out = Dense(5, activation='softmax')(x)



complete_model = Model(inp, out)



complete_model.compile(optimizer='adam', loss='categorical_crossentropy')

complete_model.summary()
# Running the model using the fit_generater method for 10 epochs

history = complete_model.fit_generator(train_generator, steps_per_epoch=115, epochs=10, validation_data=valid_generator, validation_steps=20, verbose=1)
# Taking the outputs of first 100 layers from trained model, leaving the first Input layer, in a list

layer_outputs = [layer.output for layer in complete_model.layers[1:100]]



# This is image of a Rose flower from our dataset. All of the visualizations in this cell are of this image.

test_image = "../input/aptos2019-blindness-detection/test_images/021c207614d6.png"



# Loading the image and converting it to a numpy array for feeding it to the model. Its important to use expand_dims since our original model takes batches of images

# as input, and here we are feeding a single image to it, so the number of dimensions should match for model input.

img = image.load_img(test_image, target_size=(512, 512))

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

    ax = plt.imshow(activations[0][0, :, :, img], cmap='gray')

    plt.xticks([])

    plt.yticks([])

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
# This is the visualization of activation maps from third convolution layer. In this layer the abstraction has increased. Filters are now able to regognise the edges

# of the flower more closely. Some filters are activating the surface texture of the image as well

fig = plt.figure(figsize=(22, 6))

for img in range(60):

    ax = fig.add_subplot(4, 15, img+1)

    ax = plt.imshow(activations[6][0, :, :, img], cmap='gray')

    plt.xticks([])

    plt.yticks([])

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
# These are activation maps from fourth convolution layer. The images have become a little blurry, because of the MaxPooling operation done just before this layer. As

# more Pooling layers are introduced the knowledge reaching the convolution layer becomes more and more abstract, which helps the complete network to finally classify

# the image properly, but visually they don't provide us with much information.

fig = plt.figure(figsize=(22, 6))

for img in range(60):

    ax = fig.add_subplot(4, 15, img+1)

    ax = plt.imshow(activations[10][0, :, :, img], cmap='gray')

    plt.xticks([])

    plt.yticks([])

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
# These are the activation maps from next convolution layer after next MaPooling layer. The images have become more blurry

fig = plt.figure(figsize=(22, 6))

for img in range(60):

    ax = fig.add_subplot(4, 15, img+1)

    ax = plt.imshow(activations[17][0, :, :, img], cmap='gray')

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
def deprocess_image(x):

    

    x -= x.mean()

    x /= (x.std() + 1e-5)

    x *= 0.1

    x += 0.5

    x = np.clip(x, 0, 1)

    x *= 255

    x = np.clip(x, 0, 255).astype('uint8')



    return x
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
#Below are the patterns to which the filters from first convolution layer get activated. As we can see these are very basic cross-sectional patterns formed by

# horizontal and vertical lines, which is what the these filters look in the input image and get activated if they find one

fig = plt.figure(figsize=(15, 12))

for img in range(30):

    ax = fig.add_subplot(5, 6, img+1)

    ax = plt.imshow(generate_pattern('conv2d_1', img))

    plt.xticks([])

    plt.yticks([])

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
img_path = "../input/aptos2019-blindness-detection/test_images/021c207614d6.png"



img = image.load_img(img_path, target_size=(512, 512))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)



preds = complete_model.predict(x)

preds
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