import numpy as np

import pandas as pd

import os

import math

import matplotlib.pyplot as plt

import cv2

import random

from plotly.subplots import make_subplots

from skimage import data

import plotly.graph_objs as go

from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Input

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from tensorflow.keras import backend as K

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping
MONET_JPG_PATH = '../input/gan-getting-started/monet_jpg/'

PHOTO_JPG_PATH = '../input/gan-getting-started/photo_jpg/'



print('Number of images in Monet directory: ', len(os.listdir(MONET_JPG_PATH)))

print('Number of images in Photo directory: ', len(os.listdir(PHOTO_JPG_PATH)))
shapes_set = set()

image_names = os.listdir(MONET_JPG_PATH)

for img_name in image_names:

    img = cv2.imread(os.path.join(MONET_JPG_PATH, img_name))

    shapes_set.add(img.shape)



print('Number of unique image shapes inside Monet directory: ', len(shapes_set))

print('Image shape sizes: ', shapes_set.pop())



shapes_set = set()

image_names = os.listdir(PHOTO_JPG_PATH)

for img_name in image_names:

    img = cv2.imread(os.path.join(PHOTO_JPG_PATH, img_name))

    shapes_set.add(img.shape)

print('Number of unique image shapes inside Photo directory: ', len(shapes_set))

print('Image shape sizes: ', shapes_set.pop())
def visualize_images(path, n_images, is_random=True, figsize=(16, 16)):

    plt.figure(figsize=figsize)

    w = int(n_images ** .5)

    h = math.ceil(n_images / w)

    

    all_names = os.listdir(path)

    image_names = all_names[:n_images]   

    if is_random:

        image_names = random.sample(all_names, n_images)

            

    for ind, image_name in enumerate(image_names):

        img = cv2.imread(os.path.join(path, image_name))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        plt.subplot(h, w, ind + 1)

        plt.imshow(img)

        plt.xticks([])

        plt.yticks([])

    

    plt.show()
visualize_images(MONET_JPG_PATH, 9)
visualize_images(PHOTO_JPG_PATH, 9)
def show_color_histogram(path):

    image_names = os.listdir(path)

    image_name = random.choice(image_names)

    img = cv2.imread(os.path.join(path, image_name))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    fig = make_subplots(1, 2)



    fig.add_trace(go.Image(z=img), 1, 1)

    for channel, color in enumerate(['red', 'green', 'blue']):

        fig.add_trace(

            go.Histogram(

                x=img[..., channel].ravel(), 

                opacity=0.5,

                marker_color=color, 

                name='%s channel' %color

            ), 1, 2)

    fig.update_layout(height=400)

    fig.show()
show_color_histogram(MONET_JPG_PATH)
show_color_histogram(PHOTO_JPG_PATH)
X = []

y = []



image_names = os.listdir(MONET_JPG_PATH)

for img_name in image_names:

    img = cv2.imread(os.path.join(MONET_JPG_PATH, img_name))

    X.append(img)

    y.append(1)



image_names = os.listdir(PHOTO_JPG_PATH)

for img_name in image_names:

    img = cv2.imread(os.path.join(PHOTO_JPG_PATH, img_name))

    X.append(img)

    y.append(0)

    

X = np.stack(X)

y = np.stack(y)
X = X.astype('float32') / 255.

y = to_categorical(y)

X, X_test, y, y_test = train_test_split(X, y, random_state=666, test_size=0.2, shuffle=True)
def recall_score(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_score(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def keras_f1_score(y_true, y_pred):

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
data_augmentation = tf.keras.Sequential([

    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),

    layers.experimental.preprocessing.RandomRotation(0.2),

    layers.experimental.preprocessing.RandomRotation(0.5),

])
def create_model():

    input_img = Input(shape=(256, 256, 3))

    x = data_augmentation (input_img)

    x = Conv2D(16, kernel_size=(3, 3), activation='elu')(x)

    x = MaxPooling2D(pool_size=(2,2))(x)

    x = BatchNormalization()(x)



    x = Conv2D(32, kernel_size=(3, 3), activation='elu')(x)

    x = MaxPooling2D(pool_size=(2,2))(x)

    x = BatchNormalization()(x)

    

    x = Conv2D(16, kernel_size=(3, 3), activation='elu')(x)

    x = MaxPooling2D(pool_size=(2,2))(x)

    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)

    

    x = Flatten()(x)

    x = Dense(128, activation='sigmoid')(x)

    out = Dense(2, activation = 'softmax')(x)



    model = tf.keras.Model(input_img, out)

    model.compile(

        loss=tf.keras.losses.binary_crossentropy, 

        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),

        metrics=[keras_f1_score]

    )

    

    return model
class_weight = {

    0: 1.,

    1: 22.

}
model = create_model()
early_stopping = EarlyStopping(patience=5, verbose=1)

model.fit(

    X, 

    y, 

    validation_split=0.2, 

    batch_size=20, 

    epochs=100,

    verbose=1, 

    class_weight=class_weight,  

    callbacks=[early_stopping]

)
preds = model.predict(X_test)

preds = np.argmax(preds, axis=1)

y_test = np.argmax(y_test, axis=1)

confusion_matrix(y_test, preds)
print('Accuracy: ', accuracy_score(y_test, preds))

print('F1-score: ', f1_score(y_test, preds))