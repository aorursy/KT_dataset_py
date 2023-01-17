import pandas as pd

import numpy as np

import keras

from keras.applications.vgg16 import VGG16

from keras.applications.vgg19 import VGG19

from keras.applications.mobilenet import MobileNet

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input, decode_predictions

from keras.layers import Dense, Activation, Flatten, merge,Input

from keras.models import Model, Sequential

from keras.utils import np_utils

from keras import backend as K

from keras.layers import *

from keras.callbacks import *

import os

import glob

import tensorflow as tf


train = pd.read_csv('/kaggle/input/image-classification-dataset-in-the-gala-event/image_auto_tagging/train.csv')

test  = pd.read_csv('/kaggle/input/image-classification-dataset-in-the-gala-event/image_auto_tagging/test.csv')



s = train['Class'].tolist()



from collections import Counter



print(Counter(s).keys()) # equals to list(set(words))

print(Counter(s).values()) # counts the elements' frequency

print(len(Counter(s).keys()))



tg_dict = {"Food":0, "misc": 1, "Attire": 2,"Decorationandsignage":3}

def label_encode(x):

    return tg_dict[x]



train['Class'] = train['Class'].apply(label_encode)



images = train['Image'].tolist()

classes = train['Class'].tolist()



features=[]

labels=[]

path = '/kaggle/input/image-classification-dataset-in-the-gala-event/image_auto_tagging/Train_Images/'

for i in range(0,5983):

  if os.path.isfile(path+str(images[i])):

    pic = image.load_img(path+str(images[i]), target_size=(224, 224))

    #print(path+str(images[i]))

    x = image.img_to_array(pic)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    features.append(x)

    labels.append(classes[i])

  else:

    print(path+str(images[i]), 'not present')

    

npfeatures = np.array(features)

print(npfeatures.shape)

img_dt = np.rollaxis(npfeatures, 1, 0)

print(img_dt.shape)

X = img_dt[0]

print(X.shape)

labels = np.array(labels)

Y = np_utils.to_categorical(labels,4)

print(Y.shape)





def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))





from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

#import efficientnet.tfkeras as efn



IMAGE_SIZE=[224,224]

pretrained_model = MobileNet(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

pretrained_model.trainable = False # tramsfer learning

    #enet = efn.EfficientNetB7(input_shape=(512, 512, 3),weights='imagenet',include_top=False)

    

model = Sequential([

        pretrained_model,

        GlobalAveragePooling2D(),

        Dense(220, activation='relu'),

        Dense(220, activation='relu'),

        

        Dense(4, activation='softmax')

    ])

es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)



model.compile(optimizer = 'adamax', loss='categorical_crossentropy', metrics=[f1_m])    

model.fit(X, Y, batch_size=32, epochs=10, validation_split=.1,callbacks=[es])

images_test = test['Image'].tolist()

test_features=[]

path_test = '/kaggle/input/image-classification-dataset-in-the-gala-event/image_auto_tagging/Test_Images/'

for i in range(0,3219):

  if os.path.isfile(path_test+str(images_test[i])):

    pic = image.load_img(path_test+str(images_test[i]), target_size=(224, 224))

    #print(path+str(images[i]))

    x = image.img_to_array(pic)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    test_features.append(x)

  else:

    print(path_test+str(images[i]), 'not present')
test_features = np.array(test_features)

print(test_features.shape)

test_features = np.rollaxis(test_features, 1, 0)

print(test_features.shape)

X_test = test_features[0]

print(X_test.shape)
preds = model.predict(X_test)
predictions=[]

for i in preds:

    predictions.append(np.argmax(i))
test['Class'] = predictions
gt_dict = dict((v,k) for k,v in tg_dict.items())



def inverse_encode(x):

    return gt_dict[x]



test['Class'] = test['Class'].apply(inverse_encode)
test.head(1)
test.to_csv('Submission.csv',header=True,index = None)