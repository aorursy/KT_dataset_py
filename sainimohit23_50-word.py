import sys

import numpy as np

import pickle

import os

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline



import cv2

import time

import itertools

import random



from sklearn.utils import shuffle



import tensorflow as tf

from keras.models import Sequential

from keras.optimizers import Adam, RMSprop

from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dropout

from keras.models import Model



from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import Concatenate

from keras.layers.core import Lambda, Flatten, Dense

from keras.initializers import glorot_uniform



from keras.engine.topology import Layer

from keras.regularizers import l2

from keras import backend as K

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
train_data = pd.read_csv("../input/50-word-dataset/new_data_pairs.csv")

img1, img2, values = train_data["image1"].values, train_data["image2"].values, train_data["label"].values
print("total number of training pairs", img1.shape)
# img1, img2, values = shuffle(img1, img2, values)

# img1 = img1[:50000]

# img2 = img2[:50000]

# values = values[:50000]
img1_neg = np.copy(img1[468168:])

img2_neg = np.copy(img2[468168:])

values_neg = np.copy(values[468168:])

img1_neg, img2_neg, values_neg = shuffle(img1_neg, img2_neg, values_neg)
img1[468168:] = img1_neg[:]

img2[468168:] = img2_neg[:]

values[468168:] = values_neg[:]
img1 = img1[:-424612]

img2 = img2[:-424612]

values = values[:-424612]

img1, img2, values = shuffle(img1, img2, values)
img1_val = img1[-10000:]

img2_val = img2[-10000:]

values_val = values[-10000:]
img1 = img1[:-10000]

img2 = img2[:-10000]

values = values[:-10000]
img_h, img_w = 155, 220
processed_image_dict = {}

for word in os.listdir("../input/50-word-dataset/structuredpaddedaugmentation/structuredPaddedAugmentation/"):

    for posNeg in os.listdir("../input/50-word-dataset/structuredpaddedaugmentation/structuredPaddedAugmentation/"+word+"/"):

        for file in os.listdir("../input/50-word-dataset/structuredpaddedaugmentation/structuredPaddedAugmentation/"+word+"/"+posNeg+"/"):

#                 print(word+"/"+posNeg+"/"+file)

                img = cv2.imread("../input/50-word-dataset/structuredpaddedaugmentation/structuredPaddedAugmentation/"+word+"/"+posNeg+"/"+file, 0)

                img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                img = cv2.resize(img, (img_w, img_h))

                img = np.array(img, dtype = np.float64)

                img /= 255

                processed_image_dict[word+"/"+posNeg+"/"+file] = img
def plot_img(img, isRGB=False):

    colMap = "gray"

    if isRGB:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        colMap = None

    my_dpi =30

    plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)

    plt.imshow(img, cmap=colMap)
plot_img(processed_image_dict['cresta/negative/416_aug_0_4707.png'])
def generate_batch(img1, img2, values, batch_size=16):

    while True:

        k = 0

        pairs=[np.zeros((batch_size, img_h, img_w, 1)) for i in range(2)]

        targets=np.zeros((batch_size,), dtype = np.float64)

        

        for i in range(img1.shape[0]):

            word1 = processed_image_dict[img1[i]]

            word2 = processed_image_dict[img2[i]]

            

            word1 = word1[..., np.newaxis]

            word2 = word2[..., np.newaxis]

            

            pairs[0][k, :, :, :] = word1

            pairs[1][k, :, :, :] = word2

            targets[k] = values[i]

            k += 1

            

            if k == batch_size:

                yield pairs, targets

                k = 0

                pairs=[np.zeros((batch_size, img_h, img_w, 1)) for i in range(2)]

                targets=np.zeros((batch_size,))
def euclidean_distance(vects):

    '''Compute Euclidean Distance between two vectors'''

    x, y = vects

    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
def eucl_dist_output_shape(shapes):

    shape1, shape2 = shapes

    return (shape1[0], 1)
def contrastive_loss(y_true, y_pred):

    '''Contrastive loss from Hadsell-et-al.'06

    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    '''

    margin = 1

    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
def create_base_network_signet(input_shape):

    '''Base Siamese Network'''

    

    seq = Sequential()

    seq.add(Conv2D(96, kernel_size=(11, 11), activation='relu', name='conv1_1', strides=4, input_shape= input_shape, 

                        init='glorot_uniform', dim_ordering='tf'))

    seq.add(BatchNormalization())

    seq.add(MaxPooling2D((3,3), strides=(2, 2)))    

    seq.add(ZeroPadding2D((2, 2), dim_ordering='tf'))

    

    seq.add(Conv2D(256, kernel_size=(5, 5), activation='relu', name='conv2_1', strides=1, init='glorot_uniform',  dim_ordering='tf'))

    seq.add(BatchNormalization())

    seq.add(MaxPooling2D((3,3), strides=(2, 2)))

    seq.add(Dropout(0.3))# added extra

    seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))

    

    seq.add(Conv2D(384, kernel_size=(3, 3), activation='relu', name='conv3_1', strides=1, init='glorot_uniform',  dim_ordering='tf'))

    seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))

    

    seq.add(Conv2D(256, kernel_size=(3, 3), activation='relu', name='conv3_2', strides=1, init='glorot_uniform', dim_ordering='tf'))    

    seq.add(MaxPooling2D((3,3), strides=(2, 2)))

#     seq.add(Dropout(0.3))# added extra

    seq.add(Flatten(name='flatten'))

    seq.add(Dense(1024, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform'))

#     seq.add(Dropout(0.3))

    

    seq.add(Dense(512, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform'))

#     seq.add(Dropout(0.3))

    

    seq.add(Dense(128, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform')) # softmax changed to relu

    

    return seq
input_shape=(img_h, img_w, 1)
# network definition

base_network = create_base_network_signet(input_shape)



input_a = Input(shape=(input_shape))

input_b = Input(shape=(input_shape))



# because we re-use the same instance `base_network`,

# the weights of the network

# will be shared across the two branches

processed_a = base_network(input_a)

processed_b = base_network(input_b)



# Compute the Euclidean distance between the two vectors in the latent space

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])



model = Model(input=[input_a, input_b], output=distance)
base_network.summary()
rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)

model.compile(loss=contrastive_loss, optimizer=rms)
callbacks = [

    EarlyStopping(patience=12, verbose=1),

    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),

    ModelCheckpoint('50WordModel.h5', verbose=1, save_weights_only=True)

]
BATCH_SIZE = 128

results = model.fit_generator(generate_batch(img1, img2, values, BATCH_SIZE),

                              steps_per_epoch = img1.shape[0]//BATCH_SIZE,

                              epochs = 20,

                              validation_data = generate_batch(img1_val, img2_val, values_val, BATCH_SIZE),

                              validation_steps = img1_val.shape[0]//BATCH_SIZE,

                              callbacks = callbacks)
# BATCH_SIZE = 128

# results = model.fit_generator(generate_batch(img1, img2, values, BATCH_SIZE),

#                               steps_per_epoch = 100,

#                               epochs = 200,

#                               validation_data = generate_batch(img1_val, img2_val, values_val, BATCH_SIZE),

#                               validation_steps = img1_val.shape[0]//BATCH_SIZE,

#                               callbacks = callbacks)
model.save_weights("50modelFinalb1.h5")
# import keras

# model = keras.models.load_model("../input/trained-models/50modelFinalb1.h5", custom_objects={'contrastive_loss': contrastive_loss})
k=14343

word1 = processed_image_dict[img1[k]]

word2 = processed_image_dict[img2[k]]



word1 = word1[..., np.newaxis]

word2 = word2[..., np.newaxis]



word1 = word1[np.newaxis, ...]

word2 = word2[np.newaxis, ...]



print(img1[k], img2[k])

print(model.predict([word1, word2]), values[k])
def compute_accuracy_roc(predictions, labels):

    '''Compute ROC accuracy with a range of thresholds on distances.

    '''

    dmax = np.max(predictions)

    dmin = np.min(predictions)

    nsame = np.sum(labels == 1)

    ndiff = np.sum(labels == 0)

   

    step = 0.01

    max_acc = 0

    best_thresh = -1

   

    for d in np.arange(0.0, 4+step, step):

        idx1 = predictions.ravel() <= d

        idx2 = predictions.ravel() > d

       

        tpr = float(np.sum(labels[idx2] == 1)) / nsame

        tnr = float(np.sum(labels[idx1] == 0)) / ndiff

        acc = 0.5 * (tpr + tnr)       

#       print ('ROC', acc, tpr, tnr)

#         print(d, tpr, tnr)

        if (acc > max_acc):

            max_acc, best_thresh = acc, d

           

    return max_acc, best_thresh
# img = cv2.imread("../input/50-word-dataset/structuredpaddedaugmentation/structuredPaddedAugmentation/android/positive/2893.png",0)

# img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# img = cv2.resize(img, (img_w, img_h))

# img = np.array(img, dtype = np.float64)

# img /= 255
from tqdm import tqdm_notebook as tqdm

pred, tr_y = [], []

for i in tqdm(range(img1.shape[0])):

    word1 = processed_image_dict[img1[i]]

    word2 = processed_image_dict[img2[i]]



    word1 = word1[..., np.newaxis]

    word2 = word2[..., np.newaxis]

    word1 = word1[np.newaxis, ...]

    word2 = word2[np.newaxis, ...]

    tr_y.append(values[i])

    pred.append(model.predict([word1, word2])[0][0])
tr_acc, threshold = compute_accuracy_roc(np.array(pred), np.array(tr_y))

tr_acc, threshold
def predict_score():

    '''Predict distance score and classify test images as Genuine or Forged'''

    test_point, test_label = next(test_gen)

    img1, img2 = test_point[0], test_point[1]

    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))

    ax1.imshow(np.squeeze(img1), cmap='gray')

    ax2.imshow(np.squeeze(img2), cmap='gray')

#     ax1.set_title('Genuine')

#     if test_label == 1:

#         ax2.set_title('Genuine')

#     else:

#         ax2.set_title('Forged')

    ax1.axis('off')

    ax2.axis('off')

    plt.show()

    result = model.predict([img1, img2])

    diff = result[0][0]

    print("Difference Score = ", diff)

    print("Actual Label = ", test_label[0])

    if diff > threshold:

        print("Its a Forged Signature")

    else:

        print("Its a Genuine Signature")
predict_score()