import os #to read and write a file

import cv2 #opencv for manipulating images

from glob import glob #finds all the pathnames matching a specified pattern

import h5py #will be helpful for storing huge amt of numerical data

import shutil #offers a high-level operations on files and collection of files

import imgaug as aug #Image augmentation. Helful for creating much more larger dataset from our input.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualising tool

import matplotlib.pyplot as plt #visualising tool

import imgaug.augmenters as iaa #Image augmentation.



from os import listdir, makedirs, getcwd, remove 

#listdr- return list containing names of the entries given in path

#make directory named path with the specified numeric mode.

#getcwd - getting current working directory

#remove- remove/delete a file path



from os.path import isfile, join, abspath, exists, isdir

#isfile - to check specified path is available in that file or not.

#join - to join one or more path.

#abspath - returns a normalised version of the path

#isdir- returns true/false if specified path is there in the directory or not.



from pathlib import Path #object oriented file system path.

from skimage.io import imread #image reading/writing

from skimage.transform import resize #resize

from keras.models import Sequential, Model, load_model #keras NN model



from keras.applications.vgg19 import VGG19, preprocess_input #VGG19

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten #layers to build NN

from keras.optimizers import Adam, SGD, RMSprop #optimizers

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split 

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix

from keras import backend as K

import tensorflow as tf





color = sns.color_palette()

%matplotlib inline

%config InlineBackend.figure_format="svg"
columns = ['path', 'label']

train_normal_df = pd.DataFrame(columns = columns)

train_normal_path = {os.path.basename(x): x for x in glob(os.path.join('..', 'input','chest-xray-pneumonia','chest_xray' ,'train', 'NORMAL' , '*.jpeg'))}



print('Scans found:', len(train_normal_path))



train_normal_df['path'] = train_normal_path.values()

train_normal_df['label'] = 0 #no pneumonia

train_normal_df.head(5)
columns = ['path', 'label']

train_pneumonia_df = pd.DataFrame(columns = columns)

train_pneumonia_path = {os.path.basename(x): x for x in glob(os.path.join('..', 'input','chest-xray-pneumonia','chest_xray' ,'train', 'PNEUMONIA' , '*.jpeg'))}



print('Scans found:', len(train_pneumonia_path))



train_pneumonia_df['path'] = train_pneumonia_path.values()

train_pneumonia_df['label'] = 1 #yes pneumonia
train_pneumonia_df.head(5)
#for valid datasets

columns = ['path', 'label']



test_normal_df = pd.DataFrame(columns = columns)

test_normal_path = {os.path.basename(x): x for x in glob(os.path.join('..', 'input','chest-xray-pneumonia','chest_xray' ,'test', 'NORMAL' , '*.jpeg'))}



test_pneumonia_df = pd.DataFrame(columns = columns)

test_pneumonia_path = {os.path.basename(x): x for x in glob(os.path.join('..', 'input','chest-xray-pneumonia','chest_xray' ,'test', 'PNEUMONIA' , '*.jpeg'))}



print('Scans found:', len(test_normal_path))

print('Scans found:', len(test_pneumonia_path))



test_pneumonia_df['path'] = test_pneumonia_path.values()

test_normal_df['path'] = test_normal_path.values()



test_pneumonia_df['label'] = 1 #yes pneumonia

test_normal_df['label'] = 0
#preparing dataset

frames = [train_normal_df, train_pneumonia_df, test_pneumonia_df, test_normal_df]

final_df = pd.concat(frames)

final_df = final_df.sample(frac = 1).reset_index(drop = True)

print('shuffling all the rows...')

final_df.head()
#creating a valid dataset with no labels.

#for valid datasets

columns = ['path', 'label']



valid_normal_df = pd.DataFrame(columns = columns)

valid_normal_path = {os.path.basename(x): x for x in glob(os.path.join('..', 'input','chest-xray-pneumonia','chest_xray' ,'val', 'NORMAL' , '*.jpeg'))}



valid_pneumonia_df = pd.DataFrame(columns = columns)

valid_pneumonia_path = {os.path.basename(x): x for x in glob(os.path.join('..', 'input','chest-xray-pneumonia','chest_xray' ,'val', 'PNEUMONIA' , '*.jpeg'))}



print('Scans found:', len(valid_normal_path))

print('Scans found:', len(valid_pneumonia_path))



valid_pneumonia_df['path'] = valid_pneumonia_path.values()

valid_normal_df['path'] = valid_normal_path.values()



frames = [valid_normal_df, valid_pneumonia_df]

pred_df = pd.concat(frames)

pred_df.head()
#train_test split

from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(final_df, test_size = 0.25, random_state = 2020)

print('train', train_df.shape[0], 'validation', valid_df.shape[0])
train_df
valid_df
print("Number of traininng samples: ", len(train_df))

print("Number of validation samples: ", len(valid_df))
# dimensions to consider for the images

img_rows, img_cols, img_channels = 224,224,3



# batch size for training  

batch_size=10



# total number of classes in the dataset

nb_classes=2
#augmentation

seq = iaa.OneOf([

    iaa.Fliplr(), 

    iaa.Affine(rotate=20), 

    iaa.Multiply((1.2, 1.5))]) 



#Fliplr- Horizontal Flips

#Affine - rotation

#Multiply - Random Brightness
def data_generator(data, batch_size, is_validation_data=False):

    # Get total number of samples in the data

    n = len(data)

    nb_batches = int(np.ceil(n/batch_size))



    # Get a numpy array of all the indices of the input data

    indices = np.arange(n)

    

    # Define two numpy arrays for containing batch data and labels

    batch_data = np.zeros((batch_size, img_rows, img_cols, img_channels), dtype=np.float32)

    batch_labels = np.zeros((batch_size, nb_classes), dtype=np.float32)

    

    while True:

        if not is_validation_data:

            # shuffle indices for the training data

            np.random.shuffle(indices)

            

        for i in range(nb_batches):

            # get the next batch 

            next_batch_indices = indices[i*batch_size:(i+1)*batch_size]

            

            # process the next batch

            for j, idx in enumerate(next_batch_indices):

                img = cv2.imread(data.iloc[idx]["path"])

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                label = data.iloc[idx]["label"]

                

                if not is_validation_data:

                    img = seq.augment_image(img)

                

                img = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)

                batch_data[j] = img

                batch_labels[j] = to_categorical(label,num_classes=nb_classes)

            

            batch_data = preprocess_input(batch_data)

            yield batch_data, batch_labels
#training data generator 

train_data_gen = data_generator(train_df, batch_size)



# validation data generator 

valid_data_gen = data_generator(valid_df, batch_size, is_validation_data=True)
def get_base_model():

    base_model = VGG19(input_shape=(img_rows, img_cols, img_channels), weights='imagenet', include_top=True)

    return base_model
# get the base model

base_model = get_base_model()



#  get the output of the second last dense layer 

base_model_output = base_model.layers[-2].output



# add new layers 

x = Dropout(0.3,name='drop2')(base_model_output)

output = Dense(2, activation='softmax', name='fc3')(x)



# define a new model 

model = Model(base_model.input, output)



# Freeze all the base model layers 

for layer in base_model.layers[:-1]:

    layer.trainable=False



# compile the model and check it 

optimizer = RMSprop(0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()
# the restore_best_weights parameter load the weights of the best iteration once the training finishes

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(patience=10, restore_best_weights=True)



# checkpoint to save model

chkpt = ModelCheckpoint(filepath="model1", save_best_only=True)



# number of training and validation steps for training and validation

nb_train_steps = int(np.ceil(len(train_df)/batch_size))

nb_valid_steps = int(np.ceil(len(valid_df)/batch_size))



# number of epochs 

nb_epochs=10
# train the model 

history1 = model.fit_generator(train_data_gen,

                               epochs=nb_epochs,

                               steps_per_epoch=nb_train_steps,

                               validation_data=valid_data_gen,

                               validation_steps=nb_valid_steps,

                               callbacks=es)
#get the training and validation accuracy

sns.set_style("darkgrid")

train_acc = history1.history['accuracy']

valid_acc = history1.history['val_accuracy']



#get the loss

train_loss = history1.history['loss']

valid_loss = history1.history['val_loss']



#get the entries

xvalues = np.arange(len(train_acc))



#visualise

f, ax = plt.subplots(1,2, figsize = (10,5))

ax[0].plot(xvalues, train_loss)

ax[0].plot(xvalues, valid_loss)

ax[0].set_title("Loss curve")

ax[0].set_xlabel("Epoch")

ax[0].set_ylabel("loss")

ax[0].legend(['train', 'validation'])



ax[1].plot(xvalues,  train_acc)

ax[1].plot(xvalues, valid_acc)

ax[1].set_title("Accuracy")

ax[1].set_xlabel("Epoch")

ax[1].set_ylabel("accuracy")

ax[1].legend(['train', 'validation'])



plt.show()
valid_loss, valid_acc = model.evaluate_generator(valid_data_gen, steps=nb_valid_steps)

print(f"Final validation accuracy: {valid_acc*100:.2f}%")