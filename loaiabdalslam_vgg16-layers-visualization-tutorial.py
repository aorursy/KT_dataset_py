



# load vgg model

from keras.applications.vgg16 import VGG16

# load the model

model = VGG16()

# summarize the model

model.summary()


# summarize filter shapes

for layer in model.layers:

    # check for convolutional layer

    print(layer.name)

    if 'conv' not in layer.name:

        continue



# summarize filters in each convolutional layer

from keras.applications.vgg16 import VGG16

from matplotlib import pyplot

# load the model

model = VGG16()

# summarize filter shapes

for layer in model.layers:

	# check for convolutional layer

	if 'conv' not in layer.name:

		continue

	# get filter weights

	filters, biases = layer.get_weights()

	print(layer.name, filters.shape)



# normalize filter values to 0-1 so we can visualize them

f_min, f_max = filters.min(), filters.max()

filters = (filters - f_min) / (f_max - f_min)

f = pyplot.figure(figsize=(16,16))

# plot first few filters

n_filters, ix = 6, 1

for i in range(n_filters):

	# get the filter

	f = filters[:, :, :, i]

	# plot each channel separately

	for j in range(3):

		# specify subplot and turn of axis

		ax = pyplot.subplot(n_filters, 3, ix)

		ax.set_xticks([])

		ax.set_yticks([])

		# plot filter channel in grayscale

		pyplot.imshow(f[:, :, j], cmap='gray')

		ix += 1

# show the figure

pyplot.show()

from keras.applications.vgg16 import VGG16

from matplotlib import pyplot

f = pyplot.figure(figsize=(16,16))



# load the model

model = VGG16()

# retrieve weights from the second hidden layer

filters, biases = model.layers[1].get_weights()

# normalize filter values to 0-1 so we can visualize them

f_min, f_max = filters.min(), filters.max()

filters = (filters - f_min) / (f_max - f_min)

# plot first few filters

n_filters, ix = 6, 1

for i in range(n_filters):

	# get the filter

	f = filters[:, :, :, i]

	# plot each channel separately

	for j in range(3):

		# specify subplot and turn of axis

		ax = pyplot.subplot(n_filters, 3, ix)

		ax.set_xticks([])

		ax.set_yticks([])

		# plot filter channel in grayscale

		pyplot.imshow(f[:, :, j], cmap='gray')

		ix += 1

# show the figure

pyplot.show()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

from tqdm import tqdm

import PIL

from PIL import Image, ImageOps

import cv2

from sklearn.utils import class_weight, shuffle

from keras.losses import binary_crossentropy, categorical_crossentropy

#from keras.applications.resnet50 import preprocess_input

from keras.applications.densenet import DenseNet121,DenseNet169

import keras.backend as K

import tensorflow as tf

from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score

from keras.utils import Sequence

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import imgaug as ia



WORKERS = 2

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")

SIZE = 300

NUM_CLASSES = 5
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
def display_samples(df, columns=4, rows=3):

    fig=plt.figure(figsize=(5*columns, 4*rows))



    for i in range(columns*rows):

        image_path = df.loc[i,'id_code']

        image_id = df.loc[i,'diagnosis']

        img = cv2.imread(f'../input/train_images/{image_path}.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        fig.add_subplot(rows, columns, i+1)

        plt.title(image_id)

        plt.imshow(img)

    

    plt.tight_layout()



display_samples(df_train)
x = df_train['id_code']

y = df_train['diagnosis']



x, y = shuffle(x, y, random_state=8)

y.hist()
y = to_categorical(y, num_classes=NUM_CLASSES)

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15,

                                                      stratify=y, random_state=8)

print(train_x.shape)

print(train_y.shape)

print(valid_x.shape)

print(valid_y.shape)
# plot feature map of first conv layer for given image

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.models import Model

from matplotlib import pyplot 

from numpy import expand_dims





f = plt.figure(figsize=(16,16))

# load the modelf = plt.figure(figsize=(10,3))

model = VGG16()

# redefine model to output right after the first hidden layer

model = Model(inputs=model.inputs, outputs=model.layers[1].output)

model.summary()

# load the image with the required shape

img = load_img(f'../input/test_images/270a532df702.png', target_size=(224, 224))

# convert the image to an array

img = img_to_array(img)

# expand dimensions so that it represents a single 'sample'

img = expand_dims(img, axis=0)

# prepare the image (e.g. scale pixel values for the vgg)

img = preprocess_input(img)

# get feature map for first hidden layer

feature_maps = model.predict(img)

# plot all 64 maps in an 8x8 squares

square = 8

ix = 1

for _ in range(square):

	for _ in range(square):

		# specify subplot and turn of axis

		ax = pyplot.subplot(square, square, ix)

		ax.set_xticks([])

		ax.set_yticks([])

		# plot filter channel in grayscale

		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='viridis')

		ix += 1

# show the figure

pyplot.show()

# visualize feature maps output from each block in the vgg model

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.models import Model

import matplotlib.pyplot as plt

from numpy import expand_dims









# load the model

model = VGG16()

# redefine model to output right after the first hidden layer

ixs = [2, 5, 9, 13, 17]

outputs = [model.layers[i].output for i in ixs]

model = Model(inputs=model.inputs, outputs=outputs)

# load the image with the required shape

# convert the image to an array

img = load_img(f'../input/test_images/270a532df702.png', target_size=(224, 224))

# convert the image to an array

img = img_to_array(img)

# expand dimensions so that it represents a single 'sample'

img = expand_dims(img, axis=0)

# prepare the image (e.g. scale pixel values for the vgg)

img = preprocess_input(img)

# get feature map for first hidden layer

feature_maps = model.predict(img)

# plot the output from each block

square = 8

for fmap in feature_maps:

	# plot all 64 maps in an 8x8 squares

	ix = 1

	for _ in range(square):

		plt.figure(figsize=(64,64))

		for _ in range(square):

           



			# specify subplot and turn of axis

			ax = pyplot.subplot(square, square, ix)

			ax.set_xticks([])

			ax.set_yticks([])

			

			# plot filter channel in grayscale

			plt.imshow(fmap[0, :, :, ix-1], cmap='viridis')

			ix += 1

	# show the figure



        

	plt.show()