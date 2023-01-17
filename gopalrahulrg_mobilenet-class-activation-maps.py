# importing library to handle files

import os

from os import path



# importing library to handle folders

import shutil



# importing library to display status bars

from tqdm.notebook import tqdm



# importing library to handle warnings

import warnings

warnings.filterwarnings("ignore")



# importing libraries to handle images

import cv2



# importing library to handle file checksums

import hashlib



# importing library to handle data structures

import pandas as pd



# importing library to handle arrays

import numpy as np



# importing library to handle randomness

import random



# importing library to display

import matplotlib.pyplot as plt



# importing library for deep learning

import tensorflow as tf



# importing library for preprocessing

from sklearn.model_selection import train_test_split
# initializing lists to store file paths

paths = []



# filtering original dataset with a smaller number of classes

an_list = ['black_bear', 'cougar', 'gray_wolf', 'bobcat']



# iteration through directories and preprocessing filepaths and filenames

for dirname, _, filenames in tqdm(os.walk('/kaggle/input')):

    for filename in filenames:

        

        fileloc = os.path.join(dirname, filename)

        

        if filename!='wildlife.h5':

            if fileloc.split(os.path.sep)[-2] in an_list:

                paths.append(fileloc)
# defining name for output directory

od = 'output_images'



# defining function to create output directory

def dir_tree(o_d):

    if path.isdir(o_d) == False:

        os.mkdir(o_d)

        print("Output image directory created")

    else:

        print("Output image directory already exists!")

        shutil.rmtree(o_d)

        os.mkdir(o_d)

        print("Fresh directory created under same name after clean-up!")
# defining a function to plot class distributions

def label_dist(class_dis):

    list_counts = np.unique(class_dis, return_counts=True)

    

    fig, ax = plt.subplots(figsize = (15,6))    



    ax.barh(np.arange(len(list_counts[0])), list_counts[1], 

            height = 0.3, align = 'center')

    

    ax.set_yticks(np.arange(len(list_counts[0])))

    ax.set_yticklabels(list_counts[0])

    ax.set_xlabel('Label Count')

    ax.set_title('Class Distribution')



    plt.show()
# output directory

dir_tree(od)



# creating lists to store data

label = []

img_data = []

out_p = []



# initializing resizing dimensions

IMAGE_DIMS = (224, 224, 3)



# exception count

e_count = 0



# iterating through image paths

for enum, imagePath in tqdm(enumerate(paths)):

    

    try:

        counter = 0

        img=cv2.imread(imagePath)

        

        img=cv2.resize(img, (IMAGE_DIMS[1], IMAGE_DIMS[0]))

            

    except Exception as e:

        counter = 1

        e_count = e_count + 1

    

    if counter==0:

        img_cat = imagePath.split(os.path.sep)[-2]

        label.append(img_cat)

        

        img_data.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        

        mod_name_temp = "im" + str(enum).zfill(4) + ".jpg"

        cv2.imwrite(os.path.join(od, mod_name_temp), img)

        out_p.append(os.path.join(od, mod_name_temp))

        

print("Number of images that could not be processed:", e_count)
# initial label distributions

print("Label distributions after preprocessing:")



label_dist(label)
# defining a dictionary to store hash values for processed files

hash_keys = dict()



# list to store replicates/multiple instances of files

replicates = []



# iterating through files

for enum, filename in enumerate(out_p):



    if os.path.isfile(filename):

        with open(filename, 'rb') as f:

            filehash = hashlib.md5(f.read()).hexdigest()

        if filehash not in hash_keys:

            hash_keys[filehash] = enum

        else:

            replicates.append((enum, hash_keys[filehash]))



print("Number of exact replicates found:", len(replicates))
# visualizing replicates

if len(replicates) > 0:



    # number of replicates to display

    num_repl = 8



    # randomly sampling replicates to be displayed

    sample_repl = random.sample(replicates, len(replicates))[0: num_repl]



    # flattening replicate indices that have been sampled

    sample_index = [sample_repl[i][j] for j in range(0, 2) for i in range(0, num_repl)] 



    # getting images for respective indices

    sample_images = [img_data[i] for i in sample_index]



    # getting sample image labels

    sample_labels = [label[i] for i in sample_index]



    # figure creation

    fig, ax = plt.subplots(2, num_repl, figsize=(15, 6))

    k = 0



    # creating subplots

    for i in range(0, 2):

        for j in range(0, num_repl):

            x_title = 'Index: ' + str(sample_index[k])

            ax[i][j].set_title(x_title, fontsize = 8)

            ax[i][j].imshow(sample_images[k])

            

            ax[i][j].set_xlabel(sample_labels[k], fontsize = 8)

            ax[i][j].set_yticklabels([])

            ax[i][j].set_xticklabels([])

            k = k+1

    

    plt.show()

    

else:

    print("No replicates to visualize")
# getting indices of replicates to be removed

del_indices = [index[0] for index in replicates]



# removing replicates

for index in tqdm(sorted(del_indices, reverse=True)):

    del out_p[index]

    del img_data[index]

    del label[index]
# replotting class distributions after image clean-up

print("Label distributions after clean-up:")



label_dist(label)
# dataframes for training, validation and test datasets

main_df = pd.DataFrame({'Path': out_p, 'Label': label}).sample(frac = 1, random_state = 10)



# splitting to create relatively small datasets to be used for model fitting

oX_train, X_test, oy_train, y_test = train_test_split(main_df['Path'], main_df['Label'], 

                                                      test_size = 0.8,

                                                      stratify = main_df['Label'])



# splitting into training and validation datasets

X_train, X_val, y_train, y_val = train_test_split(oX_train, oy_train, 

                                                  test_size = 0.2,

                                                  stratify = oy_train)

# train dataframe

train_df = pd.DataFrame({'Path': X_train, 'Label': y_train})



# validation dataframe

val_df = pd.DataFrame({'Path': X_val, 'Label': y_val})



# test dataframe

test_df = pd.DataFrame({'Path': X_test, 'Label': y_test})
# loading preprocessing function

prep_func = tf.keras.applications.mobilenet.preprocess_input 

        

# importing pretrained model

mnet_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape = IMAGE_DIMS,

                                                           include_top = False, weights = 'imagenet')

         

# freezing layers in pretrained model

for i, layer in enumerate(mnet_model.layers):

    layer.trainable = False



# training generator without any augmentation

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = prep_func)  



# validation/testing generator without any augmentation

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = prep_func)



# batch size for training

train_bs = 16



# loading training data in batches

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df, x_col="Path",

                                                    y_col="Label", target_size=(IMAGE_DIMS[1], 

                                                                                IMAGE_DIMS[0]),

                                                    batch_size=train_bs, 

                                                    class_mode='sparse')



# batch size for validation

val_bs = 8



# loading validation data in batches

val_generator = val_datagen.flow_from_dataframe(dataframe=val_df, x_col="Path",

                                                y_col="Label", target_size=(IMAGE_DIMS[1], 

                                                                            IMAGE_DIMS[0]),

                                                batch_size=val_bs, 

                                                class_mode='sparse')
# visualization of images from generator

train_sample_images = [next(train_generator) for i in range(18)]

train_sample_labels = [gen[1][0] for gen in train_sample_images]



# inverting encoded labels to display beside corresponding images

inv_labels = {v: k for k, v in (train_generator.class_indices).items()}

keys = list(inv_labels.keys())



coded_label = []



# storing labels to be displayed

for train_sample_label in train_sample_labels:

    for key in keys:

        if train_sample_label == key:

            coded_label.append(inv_labels.get(key))



fig, ax = plt.subplots(3,6, figsize=(15, 10))



k = 0



# creating subplots of images

for i in range(3):

    for j in range(6):

        ax[i][j].set_title(coded_label[k], fontsize = 8)

        ax[i][j].imshow(np.array(train_sample_images[k][0][0]))

        

        ax[i][j].set_yticklabels([])

        ax[i][j].set_xticklabels([])

        k = k + 1



plt.show()
# defining a sequential model to learn 

model = mnet_model.layers[-3].output



# adding pretrained model

model = tf.keras.layers.GlobalAveragePooling2D()(model)



model = tf.keras.layers.Dense(len(np.unique(y_train)), activation=tf.nn.softmax)(model)



# define a new model 

clf_model = tf.keras.Model(mnet_model.input, model)



clf_model.summary()
# compiling the model

clf_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])



# training and validation steps for model fitting

train_steps = np.ceil(X_train.shape[0]/train_bs)

val_steps = np.ceil(X_val.shape[0]/val_bs)



# creating a callback to stop training

class es_myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        

        # setting an accuracy threshold

        AC_TH = 0.90

        

        # checking if threshold has been reached

        if(logs.get('val_accuracy') > AC_TH and logs.get('accuracy') > AC_TH):

            print("\nReached set accuracy threshold on training and validation!!")

            self.model.stop_training = True



# instantiating a callback object

earlystop = es_myCallback()



# list of callbacks

callbacks_list = [earlystop]



# training

history = clf_model.fit_generator(train_generator, steps_per_epoch=train_steps,

                                  validation_data = val_generator, epochs = 10,

                                  validation_steps = val_steps, 

                                  callbacks = callbacks_list, verbose = 1)
# class indices for validation generator

print("Class mappings of validation generator:", val_generator.class_indices)



# filtering test dataframe for new class

cam_df = test_df[test_df.Label == 'bobcat']



# getting images and labels for test class

cam_paths = cam_df['Path']

cam_labels = cam_df['Label'].apply(lambda x: 1)
# batch size for testing

test_bs = 32



# steps for testing

test_steps = np.ceil(X_test.shape[0]/test_bs)



# loading validation data in batches

test_generator = val_datagen.flow_from_dataframe(dataframe=cam_df, x_col="Path",

                                                 target_size=(IMAGE_DIMS[1], 

                                                              IMAGE_DIMS[0]),

                                                 batch_size=test_bs, 

                                                 class_mode=None, shuffle = False)



# getting label probabilities from test set

cam_probs = clf_model.predict_generator(test_generator, verbose = 1)



# getting labels for target class from test set

cam_preds = np.argmax(cam_probs, axis=-1)
# getting missed detections for target class from test set

missdet_target = cam_labels.shape[0] - np.unique(cam_preds, 

                                                 return_counts=True)[1][1]



print("Number of missed detections:", missdet_target)

                                                                                   

# getting recall value for target set from test set

recall_target = round(np.unique(cam_preds, 

                                return_counts=True)[1][1]/cam_labels.shape[0],2)



print("Recall value for target class:", recall_target)
# dataframe of predictions for target class

cam_corr_preds = pd.DataFrame({"Predicted_Class": cam_preds})



# randomly sampling correct predictions for target class

cam_corr_preds = cam_corr_preds[cam_corr_preds['Predicted_Class']==1].sample(n = missdet_target)



# getting indices of rows with correct predictions

corr_indices = list(cam_corr_preds.index)
# getting class weights for last layer in model

class_weights = clf_model.layers[-1].get_weights()[0]



# commening process of getting class weights for last convolutional layer by first defining it

final_conv_layer = clf_model.layers[-3]



# defining a backend function to get outputs for various layers in the model

get_output = tf.keras.backend.function([clf_model.layers[0].input], 

                                       [final_conv_layer.output])



# target indices

cam_len = np.arange(cam_df.shape[0])



# iterating through files and labels for target class

for cam_path, cam_label, cam_prob, cam_pred, ind  in zip(cam_paths, cam_labels, 

                                                         cam_probs, cam_preds, cam_len): 

    

    # loading and reading images

    image_loaded = cv2.imread(cam_path)

    image_loaded = cv2.cvtColor(image_loaded, cv2.COLOR_BGR2RGB)

    image_loaded = np.asarray(image_loaded)

    

    # preprocessing images to make predictions using the model

    prep_loaded = prep_func(image_loaded)

    prep_loaded = np.expand_dims(prep_loaded, axis=0)

    

    # getting outputs for each target file

    [conv_outputs] = get_output(prep_loaded)

    conv_outputs = conv_outputs[0, :, :, :]

    

    # initializing a matrix to store class activation map

    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])

    

    # iterating through weights and adding them to activation map

    for index, weight in enumerate(class_weights[:, cam_label]):

        cam += weight * conv_outputs[:, :, index]

    

    # normalizing activation map

    cam = np.maximum(cam, 0)

    cam /= np.max(cam)

    

    # postprocessing heatmap

    heatmap = cv2.resize(cam, (IMAGE_DIMS[1], IMAGE_DIMS[0]))

    heatmap = heatmap * 255

    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    

    # superimposing heatmap and image

    img = heatmap * 0.5 + image_loaded

    img = np.clip(img, 0, 255).astype(np.uint8)

    

    # displaying plots only for those predictions that do not match actual label

    if cam_label != cam_pred or ind in corr_indices:

        

        print("Test Set Target Index:", ind)

        

        if cam_label != cam_pred:

            print("Misclassified Image")

            

        elif ind in corr_indices:

            print("Correctly Classified Image")

        

        fig, ax = plt.subplots(1, 3, figsize=(15, 10))

        

        # plotting original image

        ax[0].imshow(image_loaded)

        ax[0].set_title('Original Image Plot', 

                        fontsize = 12)

        ax[0].set_xlabel(f'True Class: {inv_labels.get(cam_label)}', 

                         fontsize = 12)

        ax[0].set_yticklabels([])

        ax[0].set_xticklabels([])

        

        # plotting activation map

        ax[1].imshow(heatmap)

        ax[1].set_title('Activation Map Plot', 

                        fontsize = 12)

        ax[1].set_xlabel(f'Predicted Class: {inv_labels.get(cam_pred)}', 

                         fontsize = 12)

        ax[1].set_yticklabels([])

        ax[1].set_xticklabels([])

        

        # plotting superimposed image

        ax[2].imshow(img)

        ax[2].set_title('Sumperimposed Image Plot', 

                        fontsize = 12)

        ax[2].set_xlabel(f'Prediction Probability: {round(np.max(cam_prob)*100,1)} %', 

                         fontsize = 12)

        ax[2].set_yticklabels([])

        ax[2].set_xticklabels([])



        plt.show()