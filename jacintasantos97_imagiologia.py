from mlxtend.plotting import plot_confusion_matrix

import matplotlib.pyplot as plt

import numpy as np

import sys

import os

import argparse



import random

from sklearn.metrics import confusion_matrix

import time

import datetime



from collections import Counter



import numpy as np

import pandas as pd



import shutil

#from tqdm import tqdm



import inspect

import gc



import re



from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

import keras



from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator



from keras import models

from keras.models import Model

from keras.models import Sequential

from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, GlobalAveragePooling1D, GlobalAveragePooling2D, Flatten, BatchNormalization, Dense





from keras.applications.inception_v3 import InceptionV3





from keras.constraints import maxnorm





from keras import optimizers

from keras.optimizers import Adam, SGD , RMSprop





from keras import backend as K

K.set_image_dim_ordering('th')





from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau



from keras.wrappers.scikit_learn import KerasClassifier









import tensorflow as tf







from IPython.display import display



import seaborn as sns



from matplotlib.pyplot import figure

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import cv2

 

np.random.seed(9)



# reset tensorflow graph tp free up memory and resource allocation 

def reset_graph(model=None):

    if model:

        try:

            del model

        except:

            return False

    

    tf.get_default_graph()

    

    K.clear_session()

    

    gc.collect()

    

    return True





# reset callbacks 

def reset_callbacks(checkpoint=None, reduce_lr=None, early_stopping=None, tensorboard=None):

    checkpoint = None

    reduce_lr = None

    early_stopping = None

    tensorboard = None
# Configure input/ output directory

# Configure training, validation, testing directory







training_dir =  '../input/data/data/train'

testing_dir =  '../input/data/data/test'







reset_graph()

reset_callbacks()
def get_fig_axs(subplot_params):

    fig, axs = plt.subplots(

        nrows=subplot_params["nrows"], ncols=subplot_params["ncols"], 

        figsize=(subplot_params["figsize_col"], subplot_params["figsize_row"]),

        dpi=subplot_params["dpi"], facecolor=subplot_params["facecolor"], 

        edgecolor=subplot_params["edgecolor"], subplot_kw=subplot_params["subplot_kw"])

        

    return fig, axs

def plot_sample_image(image_file_paths, plot_params, subplot_params, update_image=True):

    fig, axs = get_fig_axs(subplot_params)



    plt.rcParams.update({'axes.titlesize': plot_params["axes.titlesize"]})

    plt.subplots_adjust(hspace=subplot_params["hspace"], wspace=subplot_params["wspace"])





    i=0

    for img_filepath in image_file_paths:

        img = cv2.imread(img_filepath, 1)

        plt.title(img_filepath.split("/")[-1])

        plt.subplot(subplot_params["nrows"], subplot_params["ncols"], i+1)

        plt.imshow(img)

        

        plt.xticks([])

        plt.yticks([])

        

        i=i+1

        

    if plot_params["update_image"] and os.path.exists(plot_params["image_file_name"]):

        os.remove(plot_params["image_file_name"])  

    if plot_params["save"]:

        fig.savefig(plot_params["image_file_name"], dpi=plot_params["dpi"])

        

    plt.tight_layout()

    plt.show()

    

def name_correct(name):

    return re.sub(r'[^a-zA-Z,:]', ' ', name).title()



def get_reset_subplot_params(nrows, ncols, dpi):

    subplot_params = {}

    subplot_params["nrows"] = nrows

    subplot_params["ncols"] = ncols



    subplot_params["figsize_col"] = subplot_params["ncols"]*2.5

    subplot_params["figsize_row"] = subplot_params["nrows"]*2.5

    subplot_params["dpi"] = dpi

    subplot_params["facecolor"] = 'w'

    subplot_params["edgecolor"] = 'k'

    subplot_params["subplot_kw"] = {'xticks': [], 'yticks': []}

    subplot_params["axes.titlesize"] = 'small'

    subplot_params["hspace"] = 0.5

    subplot_params["wspace"] = 0.3

    

    return subplot_params



def get_reset_plot_params(figsize=(15, 5), title="", xlabel ="", ylabel="", legends=[], title_fontsize = 18, label_fontsize = 14, image_file_name="", save = False, dpi=100, update_image=True):

    plot_params = {}

    

    plot_params["figsize"] = figsize

    

    plot_params["title"] = title

    

    plot_params["xlabel"] = xlabel

    plot_params["ylabel"] = ylabel

    

    plot_params["legends"] = legends 

    

    plot_params["title_fontsize"] = title_fontsize

    plot_params["axes.titlesize"] = "small"

    plot_params["label_fontsize"] = label_fontsize

    

    plot_params["image_file_name"] = image_file_name

    plot_params["save"] = save

    plot_params["update_image"] = update_image

    

    plot_params["subplot"] = None

    return plot_params



def select_image_by_category(image_dir, image_count_per_category):

    classes = os.listdir(image_dir)

    class_count = len(classes)



    image_file_paths = {}

    

    for i in range(class_count):

        subdir_path = image_dir+"/"+classes[i]

        subdir_files = os.listdir(subdir_path)



        subdir_file_count = len(subdir_files)



        subdir_file_mem = {}

        

        subdir_file_index = -1

        

        image_file_paths[classes[i]] = []

        

        for j in range(image_count_per_category):

            while subdir_file_index in subdir_file_mem:

                subdir_file_index = random.randint(0, subdir_file_count-1)

                

            subdir_file_mem[subdir_file_index] = 1

            

            subdir_file_name = subdir_files[subdir_file_index]

            subdir_file_path = subdir_path+ "/" + subdir_file_name



            image_file_paths[classes[i]].append(subdir_file_path)

            

    return image_file_paths

def show_class_sample_images(directory, image_count_per_category=5, save=False, dpi=100, update_image=False):

    class_count = len(os.listdir(directory))

    print("Number of Class: ", class_count)

   

    sample_img_by_class = select_image_by_category(directory, image_count_per_category)

    

    for class_name in sample_img_by_class:

        plot_params = get_reset_plot_params(image_file_name="img.png", save = save, dpi=dpi, update_image=update_image)

        subplot_params = get_reset_subplot_params(nrows=1, ncols=image_count_per_category, dpi=dpi)

        print("%s%s%s"%("-"*55, name_correct(class_name), "-"*55))

        plot_sample_image(sample_img_by_class[class_name], plot_params, subplot_params)

      

   

    

show_class_sample_images(training_dir, image_count_per_category=5, save=False, dpi=100, update_image=False)
# show barplot

def bar_plot(x, y, plot_property):

    if plot_property['subplot']:

        plt.subplot(plot_property['subplot'])

    sns.barplot(x=x, y=y)

    plt.title(plot_property['title'], fontsize=plot_property['title_fontsize'])

    plt.xlabel(plot_property['xlabel'], fontsize=plot_property['label_fontsize'])

    plt.ylabel(plot_property['ylabel'], fontsize=plot_property['label_fontsize'])

    plt.xticks(range(len(x)), x)

    

# count number of files in each subdirectory of a directory

def subdirectory_file_count(master_directory):

    subdirectories = os.listdir(master_directory)

    subdirectory_count = len(subdirectories)



    subdirectory_names = []

    subdirectory_file_counts = []



    for subdirectory in subdirectories:

        current_directory = os.path.join(master_directory, subdirectory)

        file_count = len(os.listdir(current_directory))

        subdirectory_names.append(subdirectory)

        subdirectory_file_counts.append(file_count)

    

    return subdirectory_names, subdirectory_file_counts

         



# show bar plot for count of labels in subdirectory of a directory

def count_bar_plot(master_directory, plot_property):

    dir_name, dir_file_count = subdirectory_file_count(master_directory)

    x = [name_correct(i) for i in dir_name]

    # x = dir_name

    y = dir_file_count

    bar_plot(x, y, plot_property)

    

def show_train_val_test(training_dir, testing_dir, plot_property):

    plt.figure(figsize=plot_property['figsize'])

    

    title = plot_property['title']

    plot_property['title'] = title + " (Training)"

    subplot_no = plot_property['subplot'] 



    count_bar_plot(training_dir, plot_property)

  

    plot_property['title'] = title + " (Testing)"

    plot_property['subplot'] = subplot_no + 1

    count_bar_plot(testing_dir, plot_property)

    

    plt.show()

    

    ##################33

plot_params = get_reset_plot_params()



plot_params['figsize'] = (18,4)



plot_params['title_fontsize'] = 13

plot_params['label_fontsize'] = 10



plot_params['title'] = "Number of Cases"



plot_params['subplot'] = 131



show_train_val_test(training_dir, testing_dir, plot_params)



from imgaug import augmenters as iaa

seq=iaa.Sequential([

    iaa.Crop(px=(0,16)), 

    iaa.GaussianBlur(sigma=(0,3.0)),

    #iaa.Add((-40, 40)),

    #iaa.ContrastNormalization((0.5, 1.5))

])



train_datagen = ImageDataGenerator(

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    rotation_range=10,

    fill_mode='nearest',

    #channel_shift_range=True,

    preprocessing_function=seq.augment_image

   )





train_generator = train_datagen.flow_from_directory(

    training_dir,

    target_size=(200, 200),

    class_mode='binary',

    batch_size=32,

    color_mode="grayscale",

    shuffle=True)



test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_directory(

    testing_dir,

    target_size=(200, 200),

    class_mode='binary',

    batch_size=32,

    color_mode="grayscale",

    shuffle = False)
from sklearn.utils import class_weight

def get_weight(y):

    class_weight_current =  class_weight.compute_class_weight('balanced', np.unique(y), y)

    return class_weight_current





class_weight = get_weight(train_generator.classes)

class_weight
reset_graph()

reset_callbacks()
def create_directory(directory_path, remove=False):

    if remove and os.path.exists(directory_path):

        try:

            shutil.rmtree(directory_path)

            os.mkdir(directory_path)

        except:

            print("Could not remove directory : ", directory_path)

            return False

    else:

        try:

            os.mkdir(directory_path)

        except:

            print("Could not create directory: ", directory_path)

            return False

        

    return True



model_dir = 'data/models'

create_directory(model_dir, remove=True)

model_file=model_dir + "{epoch:02d}-val_acc-{val_acc:.2f}-val_loss-{val_loss:.2f}.hdf5"



checkpoint = ModelCheckpoint(

    "{epoch:02d}-val_acc-{val_acc:.2f}-val_loss-{val_loss:.2f}.hdf5", 

    monitor='val_acc', 

    save_best_only=True)









early_stopping = EarlyStopping(

    monitor='val_loss',

    patience=10,

    verbose=1,

    restore_best_weights=True)



reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    patience=3,

    cooldown=2,

    min_lr=0.0000000001,

    verbose=1)



callbacks = [checkpoint, reduce_lr,early_stopping]



from keras.applications import mobilenetv2, densenet, Xception, NASNetMobile

def get_conv_model():





    model = Sequential()

    model.add(Conv2D(10, (3, 3), activation='relu', padding="same", input_shape=(1,200,200)))  

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(10, (3, 3), activation='relu', padding="same"))

    model.add(Conv2D(20, (3, 3), padding="same", activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(20, (3, 3), padding="valid", activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(20, (3, 3), padding="valid", activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(980, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu'))

    model.add(Dense(1 , activation='sigmoid'))





    print(model.summary())

    

    return model
print("Getting Base Model")

model = get_conv_model()

#from keras.engine.saving import load_model

#model= load_model(r"../input/data/data/models10-val_acc-0.92-val_loss-0.24.hdf5")
print("Starting Trainning Model")







    

steps_per_epoch=train_generator.n//train_generator.batch_size#len(train_generator)/train_generator.batch_size

validation_steps=test_generator.n//test_generator.batch_size#len(test_generator)/train_generator.batch_size



#STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

#STEP_SIZE_VALID=test_generator.n//test_generator.batch_size







#lr = 0.00001

# optimizer=optimizers.Adam(lr=lr)







#optimizer=optimizers.Adam(lr=lr)

loss='binary_crossentropy'

metrics=['accuracy']

epochs = 25











model.compile(optimizer='adam', loss=loss, metrics=metrics) #sgd



history = model.fit_generator(

    train_generator,

    steps_per_epoch = steps_per_epoch,

    workers=2,

    epochs=epochs,

    #verbose=2,

    callbacks=callbacks,

    validation_data=test_generator,

    validation_steps=validation_steps,

    class_weight=class_weight

    )



print("Completed Model Trainning")
xlabel = 'Epoch'

legends = ['Training', 'Validation']



ylim_pad = [0.01, 0.1]





plt.figure(figsize=(15, 5))



# Plot training & validation Accuracy values



y1 = history.history['acc']

y2 = history.history['val_acc']



min_y = min(min(y1), min(y2))-ylim_pad[0]

max_y = max(max(y1), max(y2))+ylim_pad[0]





plt.subplot(121)



plt.plot(y1)

plt.plot(y2)



plt.title('Model Accuracy', fontsize=17)

plt.xlabel(xlabel, fontsize=15)

plt.ylabel('Accuracy', fontsize=15)

plt.ylim(min_y, max_y)

plt.legend(legends, loc='upper left')

plt.grid()



                         

# Plot training & validation loss values

    

y1 = history.history['loss']

y2 = history.history['val_loss']



min_y = min(min(y1), min(y2))-ylim_pad[1]

max_y = max(max(y1), max(y2))+ylim_pad[1]

    

    

plt.subplot(122)



plt.plot(y1)

plt.plot(y2)



plt.title('Model Loss', fontsize=17)

plt.xlabel(xlabel, fontsize=15)

plt.ylabel('Loss', fontsize=15)

plt.ylim(min_y, max_y)

plt.legend(legends, loc='upper left')

plt.grid()

                         

plt.show()