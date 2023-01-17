# importing library to handle files

import os



# importing libray to handle status bars

from tqdm.notebook import tqdm



# import libray to ignore warnings

import warnings

warnings.filterwarnings("ignore")



# importing library to process data structures

import pandas as pd



# importing library to deal with numeric arrays

import numpy as np



# importing deep learning library

import tensorflow as tf



# importing library for preprocessing

from sklearn.model_selection import train_test_split
# initializing lists to store file paths for training and validation

img_paths = []



# importing libraries to store label references

labels = []



# iterating through directories

for dirname, _, filenames in tqdm(os.walk('/kaggle/input')):

    for filename in filenames:

        

        path = os.path.join(dirname, filename)

        

        if '.jpg' in path:

        

            img_paths.append(path)

            labels.append(path.split(os.path.sep)[-2])
# dataframes for training, validation and test datasets

main_df = pd.DataFrame({'Path': img_paths, 'Label': labels}).sample(frac = 1,

                                                                    random_state = 10)



oX_train, X_test, oy_train, y_test = train_test_split(main_df['Path'], main_df['Label'], test_size = 0.2,

                                                      stratify = main_df['Label'], 

                                                      shuffle = True, random_state = 20)



X_train, X_val, y_train, y_val = train_test_split(oX_train, oy_train, test_size = 0.2,

                                                  stratify = oy_train, 

                                                  shuffle = True, random_state = 40)



# train dataframe

train_df = pd.DataFrame({'Path': X_train, 'Label': y_train})



# validation dataframe

val_df = pd.DataFrame({'Path': X_val, 'Label': y_val})



# test dataframe

test_df = pd.DataFrame({'Path': X_test, 'Label': y_test})
# setting image dimensions

IMAGE_DIMS = (224, 224, 3)



# loading preprocessing function

prep_func = tf.keras.applications.vgg16.preprocess_input 

        

# importing pretrained model

vgg_model = tf.keras.applications.vgg16.VGG16(input_shape = IMAGE_DIMS,

                                              include_top = False, weights = 'imagenet')

        

# freezing layers in pretrained model

for layer in vgg_model.layers:

    layer.trainable = False



# training generator for augmentation

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = prep_func,

                                                                rotation_range = 10,  

                                                                zoom_range = 0.1, width_shift_range = 0.1,  

                                                                height_shift_range = 0.1, 

                                                                horizontal_flip = True, 

                                                                vertical_flip = True)  



# validation/testing generator for augmentation

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = prep_func)



# batch size for training

train_bs = 256



# loading training data in batches

train_generator = train_datagen.flow_from_dataframe(dataframe = train_df, x_col = "Path",

                                                    y_col = "Label", target_size = (IMAGE_DIMS[1], 

                                                                                    IMAGE_DIMS[0]),

                                                    batch_size = train_bs, 

                                                    class_mode = 'binary')



# batch size for validation

val_bs = 128



# loading validation data in batches

val_generator = val_datagen.flow_from_dataframe(dataframe = val_df, x_col="Path",

                                                y_col = "Label", target_size = (IMAGE_DIMS[1], 

                                                                                IMAGE_DIMS[0]),

                                                batch_size = val_bs, 

                                                class_mode = 'binary',

                                                shuffle = False)



# batch size for testing

test_bs = 160

    

# loading test data in batches

test_generator = val_datagen.flow_from_dataframe(dataframe = test_df, x_col = "Path",

                                                 y_col = "Label", target_size = (IMAGE_DIMS[1], 

                                                                                 IMAGE_DIMS[0]),

                                                 batch_size = test_bs, 

                                                 class_mode = 'binary',

                                                 shuffle = False)
# defining a sequential model to learn 

clf_model = tf.keras.Sequential()



# adding pretrained model

clf_model.add(vgg_model)



# using global average pooling instead of flatten and global max pooling

clf_model.add(tf.keras.layers.GlobalAveragePooling2D())



clf_model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))

clf_model.add(tf.keras.layers.Dropout(0.3))



clf_model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))

clf_model.add(tf.keras.layers.Dropout(0.3))



clf_model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))



# model summary

clf_model.summary()
# calculating steps for train, validation and testing

steps_train = np.ceil(train_df.shape[0]/train_bs)

steps_val = np.ceil(val_df.shape[0]/val_bs)

steps_test = np.ceil(test_df.shape[0]/test_bs)



print("Steps for training:", str(steps_train) + ',', "validation:", str(steps_val) + ',', 

      "testing:", str(steps_test))
# compiling the model

clf_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy',

                  metrics=['accuracy'])



# training

history = clf_model.fit_generator(train_generator, steps_per_epoch = steps_train,

                                  validation_data = val_generator, epochs = 1,

                                  validation_steps = steps_val, verbose = 1)
# evaluation

clf_eval = clf_model.evaluate_generator(test_generator, steps = steps_test, 

                                        verbose = 1)
# getting accuracy from evaluation

print("Accuracy on Test:", clf_eval[1])