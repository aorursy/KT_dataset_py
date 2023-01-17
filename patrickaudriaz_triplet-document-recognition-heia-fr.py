import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

import sys

import gc

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import cv2

import seaborn as sns

from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import LabelBinarizer

import PIL

from PIL import Image

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from sklearn.metrics import roc_curve

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.patheffects as PathEffects



from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, concatenate

from keras.models import Model, Sequential

from keras.regularizers import l2

from keras import backend as K

from keras.optimizers import SGD,Adam

from keras.losses import binary_crossentropy

import keras

from keras import layers

from keras import metrics

from keras.models import load_model

from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D, Dropout

from keras import optimizers

from keras import models

from keras.models import Sequential

from keras import preprocessing

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.applications import VGG16

from keras.utils import plot_model

from sklearn.manifold import TSNE

from itertools import permutations
img_size = 224

batch_size = 32

epochs = 100

train_size = 0.7

val_size = 0.2

test_size = 0.1

seed = 4321

channels = 3

learning_rate = 0.0001
d = '../input/tobacco3482-jpg/Tobacco3482-jpg/'

PATH = '../'



classes = (os.listdir(d))



paths = [os.path.join(d, o) for o in os.listdir(d) 

                    if os.path.isdir(os.path.join(d,o))]



nbEntries = []



for i in range(len(classes)):

    nbEntries.append(len(os.listdir(paths[i])))



#################################################

    

print(classes)

print(nbEntries)



df = pd.DataFrame({'classes':classes, 'entries':nbEntries})

ax = df.sort_values(by='entries', ascending=True).plot.bar(x='classes', y='entries', color='cornflowerblue',legend=False, figsize=(12,8))

ax.set_title('Tobacco3482 classes Distribution')

ax.set_ylabel("# entries")

for p in ax.patches:

    ax.annotate(str(p.get_height()), xy=(p.get_x(), p.get_height()-30))
total_set = []

total_labels = []





for root, dirs, files in os.walk(d):

    for file in files:

        if file.endswith(".jpg"):

            path = os.path.join(root, file)

            total_set.append(path)

            total_labels.append(root.split(os.path.sep)[-1])

            



# Return image class based on list entry (path)         

def getClass(img):

    return img.split(os.path.sep)[-2]





print(total_set[0])

print('GetClass : ', getClass(total_set[0]))

print('Label : ', total_labels[0])
random.Random(seed).shuffle(total_set)



for ima in total_set[0:3] :

    print(ima)

    img = mpimg.imread(ima)

    plt.figure(figsize=(7,7))

    imgplot = plt.imshow(img, cmap="gray")

    plt.show()

# Get data and separate it in sets

total_len = len(total_set)

index = 0



train_set = []

train_label = []



val_set = []

val_label = []



test_set = []

test_label = []





for i in total_set[0: int(total_len*train_size)] :

    train_set.append(i)

    train_label.append(getClass(i))

    

index = int(total_len*train_size)+1

    

for i in total_set[index: int(index + total_len*val_size)] :

    val_set.append(i)

    val_label.append(getClass(i))

    

index = int(index + total_len*val_size)+1 



for i in total_set[index: total_len] :

    test_set.append(i)

    test_label.append(getClass(i))



print(val_set[200])

print(val_label[200])

#################################################

# TRAIN SET

instances = [0] * len(classes)

for index, val in enumerate(classes) :

    for e in train_set :

        if(val == getClass(e)) :

            instances[index] += 1



            

df = pd.DataFrame({'classes':classes, 'entries':instances})

ax = df.sort_values(by='entries', ascending=True).plot.bar(x='classes', y='entries', color='cornflowerblue',legend=False, figsize=(12,8))

ax.set_title('Tobacco3482 TRAIN SET Distribution')

ax.set_ylabel("# entries")

for p in ax.patches:

    ax.annotate(str(p.get_height()), xy=(p.get_x(), p.get_height()-20))



#################################################

# VAL SET

instances = [0] * len(classes)

for index, val in enumerate(classes) :

    for e in val_set :

        if(val == getClass(e)) :

            instances[index] += 1



            

df = pd.DataFrame({'classes':classes, 'entries':instances})

ax = df.sort_values(by='entries', ascending=True).plot.bar(x='classes', y='entries', color='cornflowerblue',legend=False, figsize=(12,8))

ax.set_title('Tobacco3482 VAL SET Distribution')

ax.set_ylabel("# entries")

for p in ax.patches:

    ax.annotate(str(p.get_height()), xy=(p.get_x(), p.get_height()-3))

    

#################################################

# TEST SET

instances = [0] * len(classes)

for index, val in enumerate(classes) :

    for e in test_set :

        if(val == getClass(e)) :

            instances[index] += 1



            

df = pd.DataFrame({'classes':classes, 'entries':instances})

ax = df.sort_values(by='entries', ascending=True).plot.bar(x='classes', y='entries', color='cornflowerblue',legend=False, figsize=(12,8))

ax.set_title('Tobacco3482 TEST SET Distribution')

ax.set_ylabel("# entries")

for p in ax.patches:

    ax.annotate(str(p.get_height()), xy=(p.get_x(), p.get_height()-8))
def process_images(img_set) : 

    processed_img = []



    for i in range(len(img_set)) :

        processed_img.append(cv2.resize(cv2.imread(img_set[i], cv2.IMREAD_COLOR), (img_size, img_size)))

    

    return processed_img

    

data_train = process_images(train_set)

data_test = process_images(test_set)

data_val = process_images(val_set)
lb = LabelBinarizer()

lb.fit(list(classes))



x_train = np.array(data_train)

y_train =lb.transform(np.array(train_label))



x_test = np.array(data_test)

y_test = lb.transform(np.array(test_label))



x_val = np.array(data_val)

y_val = lb.transform(np.array(val_label))



print("train shape : ", x_train.shape)

print(y_train.shape)

print("test shape : ", x_test.shape)

print(y_test.shape)

print("valdiation shape : ", x_val.shape)

print(y_val.shape)





for i in range(3) :

    plt.figure(figsize=(6,6))

    imgplot = plt.imshow(x_train[i])



print(train_label[0])

print(y_train[0])

# find class with binarizer

print(lb.classes_)
# Define our own plot function

def scatter(x, labels, subtitle=None):

    # We choose a color palette with seaborn.

    palette = np.array(sns.color_palette("hls", 10))



    # We create a scatter plot.

    f = plt.figure(figsize=(8, 8))

    ax = plt.subplot(aspect='equal')

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,

                    c=palette[labels.astype(np.int)])

    plt.xlim(-25, 25)

    plt.ylim(-25, 25)

    ax.axis('off')

    ax.axis('tight')



    # We add the labels for each digit.

    txts = []

    for i in range(10):

        # Position of each label.

        xtext, ytext = np.median(x[labels == i, :], axis=0)

        txt = ax.text(xtext, ytext, str(i), fontsize=24)

        txt.set_path_effects([

            PathEffects.Stroke(linewidth=5, foreground="w"),

            PathEffects.Normal()])

        txts.append(txt)

        

    if subtitle != None:

        plt.suptitle(subtitle)

        

    plt.savefig(subtitle)
x_train_flat = x_train.reshape(-1,784)

x_test_flat = x_test.reshape(-1,784)



print(x_train_flat.shape)

print(x_test_flat.shape)

tsne = TSNE()

train_tsne_embeds = tsne.fit_transform(x_train_flat[:20])

scatter(train_tsne_embeds, y_train[:20], "Samples from Training Data")



eval_tsne_embeds = tsne.fit_transform(x_test_flat[:20])

scatter(eval_tsne_embeds, y_test[:20], "Samples from Validation Data")