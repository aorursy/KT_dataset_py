# -*- coding: utf-8 -*-

# I don't like warnings, especially user warnings at all!
import warnings
warnings.filterwarnings('ignore')
# Import some packages that we require
import os
import glob
import umap
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential, Model
from keras.applications import vgg16
from keras.applications import resnet50
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from skimage.io import imread, imshow
from skimage.transform import resize
import imgaug as ia
from imgaug import augmenters as iaa
from keras import backend as K
import tensorflow as tf
from collections import defaultdict, Counter
print(os.listdir("../input"))
# For plotting within the notebook
%matplotlib inline

# Graphics in SVG format are more sharp and legible
%config InlineBackend.figure_format = 'svg'

# seaborn color palette 
color = sns.color_palette()

# For REPRODUCIBILITY
seed = 111
np.random.seed(seed)
tf.set_random_seed(seed)
# Let us define some paths first
input_path = Path("../input")

# Path to training images and corresponding labels provided as numpy arrays
kmnist_train_images_path = input_path/"kmnist-train-imgs.npz"
kmnist_train_labels_path = input_path/"kmnist-train-labels.npz"

# Path to the test images and corresponding labels
kmnist_test_images_path = input_path/"kmnist-test-imgs.npz"
kmnist_test_labels_path = input_path/"kmnist-test-labels.npz"
# Load the training data from the corresponding npz files
kmnist_train_images = np.load(kmnist_train_images_path)['arr_0']
kmnist_train_labels = np.load(kmnist_train_labels_path)['arr_0']

# Load the test data from the corresponding npz files
kmnist_test_images = np.load(kmnist_test_images_path)['arr_0']
kmnist_test_labels = np.load(kmnist_test_labels_path)['arr_0']

print(f"Number of training samples: {len(kmnist_train_images)} where each sample is of size: {kmnist_train_images.shape[1:]}")
print(f"Number of test samples: {len(kmnist_test_images)} where each sample is of size: {kmnist_test_images.shape[1:]}")
# Get the unique labels
labels = np.unique(kmnist_train_labels)

# Get the frequency count for each label
frequency_count = np.bincount(kmnist_train_labels)

# Visualize 
plt.figure(figsize=(10,5))
sns.barplot(x=labels, y=frequency_count);
plt.title("Distribution of labels in KMNIST training data", fontsize=16)
plt.xlabel("Labels", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.show()
# Let's see how the images for different labels look like
random_samples = []
for i in range(10):
    samples = kmnist_train_images[np.where(kmnist_train_labels==i)][:3]
    random_samples.append(samples)

# Converting list into a numpy array
random_samples = np.array(random_samples)

# Visualize the samples
f, ax = plt.subplots(10,3, figsize=(10,20))
for i, j in enumerate(random_samples):
    ax[i, 0].imshow(random_samples[i][0,:,:], cmap='gray')
    ax[i, 1].imshow(random_samples[i][1,:,:], cmap='gray')
    ax[i, 2].imshow(random_samples[i][2,:,:], cmap='gray')
    
    ax[i,0].set_title(str(i))
    ax[i,0].axis('off')
    ax[i,0].set_aspect('equal')
    
    ax[i,1].set_title(str(i))
    ax[i,1].axis('off')
    ax[i,1].set_aspect('equal')
    
    ax[i,2].set_title(str(i))
    ax[i,2].axis('off')
    ax[i,2].set_aspect('equal')
plt.show()
# Labels mapping
labels_dict = dict([(0, u"\u304A"), (1, u"\u304D"), (2, u"\u3059"), (3, u"\u3064"),
                    (4, u"\u306A"), (5, u"\u306F"), (6, u"\u307E"), (7, u"\u3084"),
                    (8, u"\u308C"), (9, u"\u3093")])

print(labels_dict)
# A handy-dandy function to get randomly sampled data 
def get_random_samples(nb_indices):
    # Choose indices randomly 
    random_indices = np.random.choice(nb_indices, size=nb_indices, replace=False)

    # Get the data corresponding to these indices
    random_train_images = kmnist_train_images[random_indices].astype(np.float32)
    random_train_images /=255.
    random_train_images = random_train_images.reshape(nb_indices, 28*28)
    random_train_labels = kmnist_train_labels[random_indices]
    labels = np.unique(random_train_labels)
    return random_indices, random_train_images, random_train_labels, labels
#Get randomly sampled data
nb_indices = 5000
random_indices, random_train_images, random_train_labels, labels = get_random_samples(nb_indices)

# Get the actual labels from the labels dictionary
labels_name = [labels_dict[x] for x in labels]

# Get a t-SNE instance
tsne = TSNE(n_components=2, random_state=seed, perplexity=30)

# Fit tsne to the data
random_train_2D = tsne.fit_transform(random_train_images)


fig = plt.figure(figsize=(10, 8))

for i, label in zip(labels, labels_name):
    sns.scatterplot(random_train_2D[random_train_labels == i, 0], 
                random_train_2D[random_train_labels == i, 1], 
                label=i, s=18)

plt.title("Visualizating KMNIST embeddings using tSNE", fontsize=16)
plt.legend()
plt.show()
# Let's try UMAP now.
nb_indices = 10000
random_indices, random_train_images, random_train_labels, labels = get_random_samples(nb_indices)

embedding = umap.UMAP(n_components=2, metric='correlation', min_dist=0.8)
random_train_2D = embedding.fit_transform(random_train_images)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111) #projection='3d')

for i, label in zip(labels, labels):
    sns.scatterplot(random_train_2D[random_train_labels == i, 0], 
                random_train_2D[random_train_labels == i, 1], 
                label=label, s=15)
plt.title("Visualiza KMNIST embeddings using UMAP ", fontsize=16)
plt.legend()
plt.show()
# A bunch of variables. The variable have the same value as given in the keras example
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# input shape
input_shape = (img_rows, img_cols, 1)
# Process the train and test data in the exact same manner as done for MNIST
x_train = kmnist_train_images.astype('float32')
x_test = kmnist_test_images.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)

# convert class vectors to binary class matrices
y_train = to_categorical(kmnist_train_labels, num_classes)
y_test = to_categorical(kmnist_test_labels, num_classes)
# Build and train the model. 
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
# Check the test loss and test accuracy 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)
