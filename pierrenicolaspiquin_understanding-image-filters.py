# Imports
import numpy as np
import pandas as pd
import os
import h5py
import matplotlib.pyplot as plt
import math
import random
#function to load data
def load_dataset():
    train_data = h5py.File('../input/happy-house-dataset/train_happy.h5', "r")
    x_train = np.array(train_data["train_set_x"][:])  
    return x_train

# Load the data
X_train = load_dataset()
# Small function to create a dummy image, basically 4 different squares
def create_dummy_image(h=64, w=64):
    img = np.zeros((h, w))
    img[:h//2,:w//2] = 255
    img[h//2:,:w//2] = 200
    img[h//2:,w//2:] = 100
    return img
# RGB to grayscale function --> filters can be applied channel per channel 
# but here we will focus only on grayscale images
def to_grayscale(img):
    h = len(img)
    w = len(img[0])
    new_img = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            new_img[i][j] = int(img[i][j][0] * 0.3 + img[i][j][1] * 0.59 + img[i][j][2] * 0.11)
    return new_img
# Let's look at some images
img1 = to_grayscale(X_train[0])
img2 = to_grayscale(X_train[1])
img3 = to_grayscale(X_train[42])
dummy_img = create_dummy_image()

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 3))
axes[0].imshow(img1, cmap='Greys_r')
axes[1].imshow(img2, cmap='Greys_r')
axes[2].imshow(img3, cmap='Greys_r')
axes[3].imshow(dummy_img, cmap='Greys_r')
plt.show()
# small function to apply a 2D filter on a 1-channel image
def apply_conv_filter(img, conv_filter):
    h = len(img)
    w = len(img[0])
    hf = len(conv_filter)
    wf = len(conv_filter[0])
    h_off = hf // 2
    w_off = wf // 2
    new_img = np.zeros((h, w))
    for i in range(h_off, h - h_off):
        for j in range(w_off, w - w_off):
            new_value = 0
            for k in range(hf):
                for l in range(wf):
                    new_value += img[i - h_off + k][j - w_off + l] * conv_filter[k][l]
            new_img[i][j] = abs(new_value)
    
    return new_img
# Let's create basic gradient filters
grad_v = [[-1, 0, 1]]
grad_h = [[-1],
          [0],
          [1]]
# Let's define a last one
grad = [[-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]]
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
axes[0][0].imshow(img1, cmap='Greys_r')
axes[1][0].imshow(apply_conv_filter(img1, grad_v), cmap='Greys_r')
axes[2][0].imshow(apply_conv_filter(img1, grad_h), cmap='Greys_r')
axes[3][0].imshow(apply_conv_filter(img1, grad), cmap='Greys_r')

axes[0][1].imshow(img2, cmap='Greys_r')
axes[1][1].imshow(apply_conv_filter(img2, grad_v), cmap='Greys_r')
axes[2][1].imshow(apply_conv_filter(img2, grad_h), cmap='Greys_r')
axes[3][1].imshow(apply_conv_filter(img2, grad), cmap='Greys_r')

axes[0][2].imshow(img3, cmap='Greys_r')
axes[1][2].imshow(apply_conv_filter(img3, grad_v), cmap='Greys_r')
axes[2][2].imshow(apply_conv_filter(img3, grad_h), cmap='Greys_r')
axes[3][2].imshow(apply_conv_filter(img3, grad), cmap='Greys_r')

axes[0][3].imshow(dummy_img, cmap='Greys_r')
axes[1][3].imshow(apply_conv_filter(dummy_img, grad_v), cmap='Greys_r')
axes[2][3].imshow(apply_conv_filter(dummy_img, grad_h), cmap='Greys_r')
axes[3][3].imshow(apply_conv_filter(dummy_img, grad), cmap='Greys_r')
plt.show()
# Box filter
box_filter = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]])

# Here, we want to normalize the filter so the pixel values of our output
# image remain in a valid range
box_filter = (1/9) * box_filter
# Gaussian filter
gauss_filter = np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]])

# Here, we want to normalize the filter so the pixel values of our output
# image remain in a valid range
gauss_filter = (1/16) * box_filter
# First, let's add some noise to the top-right corner of the dummy image
for i in range(32):
    for j in range(32):
        dummy_img[i][32+j] = random.randint(0, 25)

# and a singular point
dummy_img[42][42] = 255
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
axes[0][0].imshow(img1, cmap='Greys_r')
axes[1][0].imshow(apply_conv_filter(img1, box_filter), cmap='Greys_r')
axes[2][0].imshow(apply_conv_filter(img1, gauss_filter), cmap='Greys_r')

axes[0][1].imshow(img2, cmap='Greys_r')
axes[1][1].imshow(apply_conv_filter(img2, box_filter), cmap='Greys_r')
axes[2][1].imshow(apply_conv_filter(img2, gauss_filter), cmap='Greys_r')

axes[0][2].imshow(img3, cmap='Greys_r')
axes[1][2].imshow(apply_conv_filter(img3, box_filter), cmap='Greys_r')
axes[2][2].imshow(apply_conv_filter(img3, gauss_filter), cmap='Greys_r')

axes[0][3].imshow(dummy_img, cmap='Greys_r')
axes[1][3].imshow(apply_conv_filter(dummy_img, box_filter), cmap='Greys_r')
axes[2][3].imshow(apply_conv_filter(dummy_img, gauss_filter), cmap='Greys_r')
plt.show()
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
tf.set_random_seed(42)
# Settings
train_path = os.path.join('..', 'input', 'digit-recognizer', 'train.csv')
raw_train_df = pd.read_csv(train_path)

# CNN model settings
size = 28
lr = 0.002
num_classes = 10

# Training settings
epochs = 10
batch_size = 128
# Utils
def parse_train_df(_train_df):
    labels = _train_df.iloc[:,0].values
    imgs = _train_df.iloc[:,1:].values
    imgs_2d = np.array([[[[float(imgs[index][i*28 + j]) / 255] for j in range(28)] for i in range(28)] for index in range(len(imgs))])
    processed_labels = [[0 for _ in range(10)] for i in range(len(labels))]
    for i in range(len(labels)):
        processed_labels[i][labels[i]] = 1
    return np.array(processed_labels), imgs_2d
# Data preprocessing
y_train_set, x_train_set = parse_train_df(raw_train_df)

x_train, x_val, y_train, y_val = train_test_split(x_train_set, y_train_set, test_size=0.20, random_state=42)
# Image visualization
n = 5
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
for i in range(n**2):
    ax = axs[i // n, i % n]
    (-x_train[i]+1)/2
    ax.imshow((-x_train[i, :, :, 0] + 1)/2, cmap=plt.cm.gray)
    ax.axis('off')
plt.tight_layout()
plt.show()
# CNN model
model = keras.Sequential()

model.add(Conv2D(6, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=(size, size, 1),
                 name='conv_1'
                ))
model.add(Conv2D(12, (3, 3), activation='relu', name='conv_2'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(lr),
              metrics=['accuracy'])

# Training
training_history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    verbose=1,
    validation_data=(x_val, y_val),
)
model.summary()
w = model.layers[0].get_weights()
filters_1_raw = w[0]
biases_1_raw = w[1]

filters_1 = [np.zeros((3, 3)) for _ in range(6)]
for i in range(3):
    for j in range(3):
        for n in range(6):
            filters_1[n][i][j] = filters_1_raw[i][j][0][n]
# Let's look at these filters
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
axes[0][0].imshow(filters_1[0], cmap='Greys_r')
axes[1][0].imshow(filters_1[1], cmap='Greys_r')
axes[2][0].imshow(filters_1[2], cmap='Greys_r')

axes[0][1].imshow(filters_1[3], cmap='Greys_r')
axes[1][1].imshow(filters_1[4], cmap='Greys_r')
axes[2][1].imshow(filters_1[5], cmap='Greys_r')
plt.show()
# Here we select an image of the MNIST dataset
mnist_img = x_train[42, :, :, 0]
plt.imshow(mnist_img, cmap='Greys_r')
layer_1_imgs = []
fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(15, 10))
for i in range(6):
    img = apply_conv_filter(mnist_img, filters_1[i])
    layer_1_imgs += [img]
    axes[i].imshow(img, cmap='Greys_r')
plt.show()
layer_to_visualize = ['conv_1', 'conv_2']

layer_outputs = [layer.output for layer in model.layers if layer.name in layer_to_visualize]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
intermediate_activations = activation_model.predict(np.expand_dims(x_train[42], axis=0))

n_layer = len(layer_to_visualize)
n_img_per_layer = 6
layer_cpt = 0

fig, axes = plt.subplots(nrows=n_layer, ncols=n_img_per_layer, figsize=(15, 10))

for layer_name, layer_activation in zip(layer_to_visualize, intermediate_activations):
    
    for j in range(n_img_per_layer):
        axes[layer_cpt][j].imshow(layer_activation[0, :, :, j], cmap='Greys_r')
        axes[layer_cpt][j].set_title("{} - map {}".format(layer_name, j))
        
    layer_cpt += 1