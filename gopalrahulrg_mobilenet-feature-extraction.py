# importing library to handle files

import os



# importing library to handle time

import time



# importing libray to handle status bars

from tqdm.notebook import tqdm



# import libray to ignore warnings

import warnings

warnings.filterwarnings("ignore")



# importing library to deal with numeric arrays

import numpy as np



# importing library to process images

import cv2



# importing deep learning library

import tensorflow as tf



# importing library for preprocessing

from sklearn.preprocessing import LabelEncoder



# importing library for plotting

import matplotlib.pyplot as plt
# initializing lists to store file paths for training and validation

train_X_path = []

val_X_path = []



# importing libraries to store label references

train_y_names = []

val_y_names = []



# iteration through directories and preprocessing filepaths and fielnames

for dirname, _, filenames in tqdm(os.walk('/kaggle/input')):

    for filename in filenames:

        

        path = os.path.join(dirname, filename)

        

        if 'training' in dirname:

            train_X_path.append(path)

            train_y_names.append(path.split(os.path.sep)[-2])

        elif 'validation' in dirname:

            val_X_path.append(path)

            val_y_names.append(path.split(os.path.sep)[-2])
# defining a function to resize images

def img_prep(features, output, dims):



    img_data = []

    labels = []



    for enum, imagePath in tqdm(enumerate(features)):

    

        try:

            counter = 0

            img=cv2.imread(imagePath)

            img=cv2.resize(img, (dims[1], dims[0]))

            

        except Exception as e:

        

            counter = 1

    

        if counter==0:

            

            label = output[enum]

            labels.append(label)

        

            img_data.append(img)

            

    return img_data, labels
# preprocessing training and validation sets

IMAGE_DIMS = (160, 160, 3)



train_X, train_y = img_prep(train_X_path, train_y_names, IMAGE_DIMS)

val_X, val_y = img_prep(val_X_path, val_y_names, IMAGE_DIMS)
# defining a function to extract features

def img_feature_extraction(x_values, pre_model):



    data = []

    

    # preprocessing and then using pretrained neural nets to extract features to be fed into Global Pooling

    for image in tqdm(x_values):

        im_toarray = tf.keras.preprocessing.image.img_to_array(image)

        

        im_toarray = np.expand_dims(image, axis=0)

        im_toarray = tf.keras.applications.mobilenet.preprocess_input(im_toarray)

        

        data.append(im_toarray)

        

    data_stack = np.vstack(data) 

    

    features = pre_model.predict(data_stack, batch_size=32)

    

    return data_stack, features
# importing pretrained MobileNet

mnet_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=IMAGE_DIMS,

                                               include_top=False, weights='imagenet')



# freezing layers

for layer in mnet_model.layers:

    layer.trainable = False
# final stage of preprocessing for both training and validation data

trainX_proc, train_features = img_feature_extraction(train_X, mnet_model)

valX_proc, val_features = img_feature_extraction(val_X, mnet_model)



# encoding labels

lb = LabelEncoder()



lb.fit(train_y)



train_y = lb.transform(train_y)

val_y = lb.transform(val_y)
# defining a sequential model to learn 

clf_model = tf.keras.Sequential()



# using global average pooling instead of flatten and global max pooling

clf_model.add(tf.keras.layers.GlobalAveragePooling2D(input_shape=train_features.shape[1:]))



clf_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

clf_model.add(tf.keras.layers.Dropout(0.3))



clf_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

clf_model.add(tf.keras.layers.Dropout(0.3))



clf_model.add(tf.keras.layers.Dense(len(np.unique(train_y)), activation=tf.nn.softmax))



clf_model.summary()
# compiling the model

EPOCHS = 100



clf_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])
# training the model



# getting time

t0 = time.time()



# fitting the model

history = clf_model.fit(train_features, train_y,

                        batch_size=32, epochs=EPOCHS,

                        verbose=0, validation_data=(val_features, val_y))



# getting new time

t1 = time.time()



# printing fitting time

print("Fitting for", EPOCHS, "epochs took", round(t1-t0,3), "seconds")
# plotting validation and accuracy history

fig = plt.figure(figsize=(15,6))



key_list = [[1, 'accuracy', 'val_accuracy', 'model_accuracy'], 

            [2, 'loss', 'val_loss', 'model_loss']]



for i, j, k, l in key_list:

    plt.subplot(1, 2, i)

    plt.plot(history.history[j])

    plt.plot(history.history[k])

    plt.title(l)

    plt.ylabel(j)

    plt.xlabel('epoch')

    plt.legend(['train', 'validation'], loc='upper left')



plt.show()