from IPython.display import Image

Image('../input/birds-transfer-learning/220px-Annas_hummingbird.jpg')
import pandas as pd

import numpy as np

import os

import cv2

import glob

import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint
top_path = '../input/100-bird-species/train'

birds = np.array(list(os.listdir(top_path)))
# pick only 20 type of birds to train on

nr_birds = 20



np.random.shuffle(birds)

birds = birds[:nr_birds]
idx_to_name = {i:x for (i,x) in enumerate(birds)}

name_to_idx = {x:i for (i,x) in enumerate(birds)}

print(idx_to_name)
def get_data_labels(path, birds, dim):

    data = []

    labels = []



    for bird in birds:

        imgs = [cv2.resize(cv2.imread(img), dim, interpolation=cv2.INTER_AREA) for img in glob.glob(path + "/" + bird + "/*.jpg")]

        for img in imgs:

            data.append(img)

            labels.append(name_to_idx[bird])

    return np.array(data), np.array(labels)
data_train, labels_train = get_data_labels('../input/100-bird-species/train', idx_to_name.values(), (224,224))

data_test, labels_test = get_data_labels('../input/100-bird-species/test', idx_to_name.values(), (224,224))

data_valid, labels_valid = get_data_labels('../input/100-bird-species/valid', idx_to_name.values(), (224,224))
def normalize(data):

    data = data / 255.0

    data = data.astype('float32')

    return data



def one_hot(labels):

    labels = np.eye(len(np.unique(labels)))[labels]

    return labels
data_train = normalize(data_train)

data_test = normalize(data_test)

data_valid = normalize(data_valid)



labels_train = one_hot(labels_train)

labels_test = one_hot(labels_test)

labels_valid = one_hot(labels_valid)
Image('../input/birds-transfer-learning/05-06_img_0027.png')
weights_path = "../input/birds-transfer-learning/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

base_model = VGG16(weights=weights_path, include_top=False, input_shape=(224, 224, 3))

base_model.summary()
# Freeze the extraction layers

for layer in base_model.layers:

    layer.trainable = False

 

base_model.summary()
Image('../input/birds-transfer-learning/05-06_img_0028.png')
from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.models import Model

 

# use “get_layer” method to save the last layer of the network

last_layer = base_model.get_layer('block5_pool')

# save the output of the last layer to be the input of the next layer

last_output = last_layer.output

 

# flatten the classifier input which is output of the last layer of VGG16 model

x = Flatten()(last_output)

 

# add our new softmax layer with 3 hidden units

x = Dense(nr_birds, activation='softmax', name='softmax')(x)
# instantiate a new_model using keras’s Model class

new_model = Model(inputs=base_model.input, outputs=x)

 

# print the new_model summary

new_model.summary()
new_model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='birds.model.hdf5', save_best_only=True)

 

history = new_model.fit(data_train, labels_train, steps_per_epoch=len(data_train),

validation_data=(data_test, labels_test), validation_steps=3, epochs=10, verbose=1, callbacks=[checkpointer])
# Analyze Training Data
plt.plot(history.history['val_accuracy'], 'b')

plt.plot(history.history['val_loss'], 'r')

plt.show()
def get_accuracy(model, data_valid, labels_valid):

    predictions = model(data_valid)

    wrong = 0

    for i, pred in enumerate(predictions):

        if( np.argmax(pred) !=  np.argmax(labels_valid[i])):

            wrong += 1

    return (len(data_valid) - wrong) / len(data_valid)
# we use the validation data to verify the accuracy

accuracy = get_accuracy(new_model, data_valid, labels_valid)

print("Accuracy:", accuracy)