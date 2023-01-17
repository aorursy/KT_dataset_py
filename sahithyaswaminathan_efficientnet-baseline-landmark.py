import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import gc #garbage collection
import cv2
import tensorflow as tf
import os
import pathlib
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
#eff_conv = EfficientNetB0(weights='imagenet', include_top=False,input_shape=(384,384,3))
# Dataset parameters:
INPUT_DIR = os.path.join('..', 'input')

DATASET_DIR = os.path.join(INPUT_DIR, 'landmark-recognition-2020')
TEST_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')
TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, 'train')
TRAIN_LABELMAP_PATH = os.path.join(DATASET_DIR, 'train.csv')
#declare image dimensions
nrows = 384
ncols = 384
channel = 3
batch_size = 64
df = pd.read_csv(TRAIN_LABELMAP_PATH)
FILENAME = glob.glob('../input/landmark-recognition-2020/train/*/*/*/*')
TRAINING_FILENAMES, VALIDATION_FILENAMES = train_test_split(FILENAME, test_size = 0.20, random_state = 42)
#[ \w-]+?(?=\.)
#from pathlib import Path
#print(Path(TRAINING_FILENAMES[0]).name)
#print(os.path.split(TRAINING_FILENAMES[0])[1][:-4])
training_groups = [os.path.split(filename)[1][:-4] for filename in TRAINING_FILENAMES]
validation_groups = [os.path.split(filename)[1][:-4] for filename in VALIDATION_FILENAMES]
y_train = df[df.id.isin(training_groups)].landmark_id
y_val = df[df.id.isin(validation_groups)].landmark_id
y_train.reset_index(drop=True, inplace=True)
y_val.reset_index(drop=True, inplace=True)
print(f'The number of unique training classes is {y_train.nunique()} of {df.landmark_id.nunique()} total classes')
print(f'The number of unique validation classes is {y_val.nunique()} of {df.landmark_id.nunique()} total classes')
print(f'Total number of training data {y_train.shape[0]}')
print(f'Total number of validation data {y_val.shape[0]}')
def read_image(list_of_images):
    X = []
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncols), interpolation = cv2.INTER_CUBIC)) 
    return X
#len(TRAINING_FILENAMES)
#sample_num = 500
Xtrain = read_image(TRAINING_FILENAMES)
y_train = y_train#.values
#sample_num = 500
Xval = read_image(VALIDATION_FILENAMES)
y_val = y_val#.values
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
#Label Encode
#y_train
#num_classes = 500#y_train.max()
num_classes = df.landmark_id.nunique()#y_train.max()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
# Swish defination
from keras.backend import sigmoid

class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish_act': SwishActivation(swish_act)})
#Model structure
# loading B0 pre-trained on ImageNet without final aka fiature extractor
model = EfficientNetB0(include_top=False, input_shape=(nrows, ncols,3), pooling='avg', weights='imagenet')

# building 2 fully connected layer 
x = model.output

x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation(swish_act)(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation(swish_act)(x)

# output layer
predictions = Dense(num_classes, activation="softmax")(x)

model_final = Model(inputs = model.input, outputs = predictions)

model_final.summary()
# ploting keras model for visualization

from keras.utils.vis_utils import plot_model
plot_model(model_final, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode="nearest",
# )

# # Note that the validation data should not be augmented!
# test_datagen = ImageDataGenerator(rescale=1.0 / 255)
Xtrain = np.array(Xtrain)
Xval = np.array(Xval)
# model.compile(
#     loss="categorical_crossentropy",
#     optimizer=optimizers.RMSprop(lr=2e-5),
#     metrics=["acc"],
# )
model_final.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.0001),
              metrics=['accuracy'])

mcp_save = ModelCheckpoint('EnetB0_GLAND_TL.h5', save_best_only=True, monitor='val_acc')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)

#print("Training....")
model_final.fit(Xtrain, y_train,
              batch_size=32,
              epochs=32,
              validation_split=0.1,
              callbacks=[mcp_save, reduce_lr],
              shuffle=True,
              verbose=1)
_, acc = model_final.evaluate(Xval, y_val)
print(acc)
sample_submission = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')
sample_submission.head()
# def inference_and_save_submission_csv(train_csv, test_directory, train_directory):
#     image_paths = [x for x in pathlib.Path(test_directory).rglob('*.jpg')]
#     test_len = len(image_paths)
#     if test_len == NUM_PUBLIC_TEST_IMAGES:
#         # Dummy submission
#         shutil.copyfile('../input/landmark-recognition-2020/sample_submission.csv', 'submission.csv')
#         return 'Job Done'
#     else:
#         test_ids, train_ids_labels_and_scores = get_similarities(train_csv, test_directory, train_directory)
#         final = generate_predictions(test_ids, train_ids_labels_and_scores)
#         return final
TEST_FILENAME = glob.glob('../input/landmark-recognition-2020/test/*/*/*/*')   
y_test = read_image(TEST_FILENAME)
scores = np.amax(model_final.predict(y_test), axis =1)
predictions = np.argmax(model_final.predict(y_test), axis=1)
#np.char.array(scores) + ' ' + np.char.array(predictions)
#map(' '.join, zip(scores, predictions))
landmarks = np.array([str(x1) +' '+ str(x2) for x1,x2 in zip(predictions, scores)])
#os.path.split(TEST_FILENAME[0])[1].split('.')
test_ids = [os.path.split(filename)[1].split('.')[0] for filename in TEST_FILENAME]
final = pd.DataFrame({'id': test_ids, 'target': predictions, 'scores': scores})
final['landmarks'] = final['target'].astype(str) + ' ' + final['scores'].astype(str)
final[['id', 'landmarks']].to_csv('submission.csv', index = False)