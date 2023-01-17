import os

import cv2

import glob

import h5py

import shutil

import imgaug as aug

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import imgaug.augmenters as iaa

from os import listdir, makedirs, getcwd, remove

from os.path import isfile, join, abspath, exists, isdir, expanduser

from pathlib import Path

from skimage.io import imread

from skimage.transform import resize

from keras.models import Sequential, Model, load_model

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten

from keras.optimizers import Adam, SGD, RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix

from keras import backend as K

import tensorflow as tf



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

from keras.callbacks import ModelCheckpoint, EarlyStopping





import math                      

import matplotlib.pyplot as plt  

import scipy                     

import cv2                      

import numpy as np               

import glob                      

import os                        

import pandas as pd              

import tensorflow as tf       

import itertools

import random

from random import shuffle      

from tqdm import tqdm            

from PIL import Image

from scipy import ndimage

from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from sklearn import metrics

%matplotlib inline



import cv2                  

import numpy as np  

from tqdm import tqdm

import os                   

from random import shuffle  

from zipfile import ZipFile

from PIL import Image



color = sns.color_palette()

%matplotlib inline

%config InlineBackend.figure_format="svg"



print('Done')
os.environ['PYTHONHASHSEED'] = '0'



seed=1234



np.random.seed(seed)



tf.set_random_seed(seed)



aug.seed(seed)
training_data = Path('../input/training/training/') 

validation_data = Path('../input/validation/validation/') 

labels_path = Path('../input/monkey_labels.txt')
labels_info = []



# Read the file

lines = labels_path.read_text().strip().splitlines()[1:]

for line in lines:

    line = line.split(',')

    line = [x.strip(' \n\t\r') for x in line]

    line[3], line[4] = int(line[3]), int(line[4])

    line = tuple(line)

    labels_info.append(line)

    

# Convert the data into a pandas dataframe

labels_info = pd.DataFrame(labels_info, columns=['Label', 'Latin Name', 'Common Name', 

                                                 'Train Images', 'Validation Images'], index=None)



cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']

labels = pd.read_csv("../input/monkey_labels.txt", names=cols, skiprows=1)

#labels

# Sneak peek 

labels_info.head(10)
labels_dict= {'n0':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'n5':5, 'n6':6, 'n7':7, 'n8':8, 'n9':9}



# map labels to common names

names_dict = dict(zip(labels_dict.values(), labels_info["Common Name"]))

print(names_dict)
train_df = []

for folder in os.listdir(training_data):

    # Define the path to the images

    imgs_path = training_data / folder

    

    # Get the list of all the images stored in that directory

    imgs = sorted(imgs_path.glob('*.jpg'))

    

    # Store each image path and corresponding label 

    for img_name in imgs:

        train_df.append((str(img_name), labels_dict[folder]))





train_df = pd.DataFrame(train_df, columns=['image', 'label'], index=None)

# shuffle the dataset 

train_df = train_df.sample(frac=1.).reset_index(drop=True)



####################################################################################################



# Creating dataframe for validation data in a similar fashion

valid_df = []

for folder in os.listdir(validation_data):

    imgs_path = validation_data / folder

    imgs = sorted(imgs_path.glob('*.jpg'))

    for img_name in imgs:

        valid_df.append((str(img_name), labels_dict[folder]))



        

valid_df = pd.DataFrame(valid_df, columns=['image', 'label'], index=None)

# shuffle the dataset 

valid_df = valid_df.sample(frac=1.).reset_index(drop=True)



####################################################################################################



# How many samples do we have in our training and validation data?

print("Number of traininng samples: ", len(train_df))

print("Number of validation samples: ", len(valid_df))



# sneak peek of the training and validation dataframes

print("\n",train_df.head(), "\n")

print("=================================================================\n")

print("\n", valid_df.head())
img_rows, img_cols, img_channels = 224,224,3



# batch size for training  

batch_size=8



# total number of classes in the dataset

nb_classes=10
# Augmentation sequence 

seq = iaa.OneOf([

    iaa.Fliplr(), # horizontal flips

    iaa.Affine(rotate=20), # roatation

    iaa.Multiply((1.2, 1.5))]) #random brightness
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

                img = cv2.imread(data.iloc[idx]["image"])

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
labels = labels['Common Name']

labels
def image_show(num_image,label):

    for i in range(num_image):

        imgdir = Path('../input/training/training/' + label)

        #print(imgdir)

        imgfile = random.choice(os.listdir(imgdir))

        #print(imgfile)

        img = cv2.imread('../input/training/training/'+ label +'/'+ imgfile)

       # print(img.shape)

        #print(label)

        plt.figure(i)

        plt.imshow(img)

        plt.title(imgfile)

    plt.show()
print(labels[4])

image_show(3,'n4')
train_dir = Path('../input/training/training/')

test_dir = Path('../input/validation/validation/')



LR = 1e-3

height=150

width=150

channels=3

seed=1337

batch_size = 64

num_classes = 10

epochs = 50

data_augmentation = True

num_predictions = 20



# Training generator

train_datagen = ImageDataGenerator(

        rescale=1./255,

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')



train_generator = train_datagen.flow_from_directory(train_dir, 

                                                    target_size=(height,width),

                                                    batch_size=batch_size,

                                                    seed=seed,

                                                    shuffle=True,

                                                    class_mode='categorical')



# Test generator

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(test_dir, 

                                                  target_size=(height,width), 

                                                  batch_size=batch_size,

                                                  seed=seed,

                                                  shuffle=False,

                                                  class_mode='categorical')



train_num = train_generator.samples

validation_num = validation_generator.samples 
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))
model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['acc'])

model.summary()
filepath=str(os.getcwd()+"/model.h5f")

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# = EarlyStopping(monitor='val_acc', patience=15)

callbacks_list = [checkpoint]#, stopper]



history = model.fit_generator(train_generator,

                              steps_per_epoch= train_num // batch_size,

                              epochs=epochs,

                              validation_data=train_generator,

                              validation_steps= validation_num // batch_size,

                              callbacks=callbacks_list, 

                              verbose = 1

                             )
train_acc = history.history['acc']

valid_acc = history.history['val_acc']



# get the loss

train_loss = history.history['loss']

valid_loss = history.history['val_loss']



# get the number of entries

xvalues = np.arange(len(train_acc))



# visualize

f,ax = plt.subplots(1,2, figsize=(10,5))

ax[0].plot(xvalues, train_loss)

ax[0].plot(xvalues, valid_loss)

ax[0].set_title("Loss curve")

ax[0].set_xlabel("Epoch")

ax[0].set_ylabel("loss")

ax[0].legend(['train', 'validation'])



ax[1].plot(xvalues, train_acc)

ax[1].plot(xvalues, valid_acc)

ax[1].set_title("Accuracy")

ax[1].set_xlabel("Epoch")

ax[1].set_ylabel("accuracy")

ax[1].legend(['train', 'validation'])



plt.show()
def plot_confusion_matrix(cm, target_names,title='Confusion matrix',cmap=None,normalize=False):

    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy

    if cmap is None:

        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float32') / cm.sum(axis=1)

        cm = np.round(cm,2)

        



    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.2f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass))

    plt.show()
from keras.models import load_model

model_trained = load_model(filepath)

# Predict the values from the validation dataset

Y_pred = model_trained.predict_generator(validation_generator, validation_num // batch_size+1)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred, axis = 1)

# Convert validation observations to one hot vectors

#Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true = validation_generator.classes,y_pred = Y_pred_classes)

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, normalize=True, target_names=labels)
print(metrics.classification_report(validation_generator.classes, Y_pred_classes,target_names=labels))