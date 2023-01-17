from glob import glob
import os
import numpy as np
import pandas as pd
import random
from skimage.io import imread

import matplotlib.pyplot as plt
%matplotlib inline

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

num_classes = 9
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
image_size = 150
nb_train_samples = 5216 # number of files in training set
batch_size = 16

EPOCHS = 40
STEPS = nb_train_samples / batch_size

## Specify the values for all arguments to data_generator_with_aug.
data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
#                                              featurewise_center = True, ## standardize pixel values across the entire dataset
#                                              featurewise_std_normalization = True, ## standardize pixel values across the entire dataset
                                             horizontal_flip = True,
                                             width_shift_range = 0.2,
                                             height_shift_range = 0.2,
                                            )
            
data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator_with_aug.flow_from_directory(
       directory = '../input/ageprev8/ageprev8/ageprev8/train/',
       target_size = (image_size, image_size),
       batch_size = batch_size,
       class_mode = 'categorical')

validation_generator = data_generator_no_aug.flow_from_directory(
       directory = '../input/ageprev8/ageprev8/ageprev8/validation/',
       target_size = (image_size, image_size),
       class_mode = 'categorical')

test_generator = data_generator_no_aug.flow_from_directory(
       directory = '../input/ageprev8/ageprev8/ageprev8/test/',
       target_size = (image_size, image_size),
       batch_size = batch_size,
       class_mode = 'categorical')

my_new_model.fit_generator(
       train_generator, # specify where model gets training data
       epochs = EPOCHS,
       steps_per_epoch=STEPS,
       validation_data=validation_generator) # specify where model gets validation data

# Evaluate the model
scores = my_new_model.evaluate_generator(test_generator)
print("\n%s: %.2f%%" % (my_new_model.metrics_names[1], scores[1]*100))
def choose_image_and_predict():
    age = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    folder_choice = (random.choice(age))
    
    age_images = glob('../input/ageprev8/ageprev8/ageprev8/validation/'+folder_choice+'/*')
    img_choice = (random.choice(age_images))

    img = load_img(img_choice, target_size=(150, 150))
    img = img_to_array(img)
    plt.imshow(img / 255.)
    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    pred_class = my_new_model.predict_classes(x)
    pred = my_new_model.predict(x)
    print("Actual class:", folder_choice)
    if pred_class[0] == 0:
        print("Predicted class: (0,2)")
        print("Likelihood:", pred[0][0].round(4))
        if pred[0][0].round(4) < 0.8:
            print("WARNING, low confidence")
    if pred_class[0] == 1:
        print("Predicted class: (3,6)")
        print("Likelihood:", pred[0][1].round(4))
        if pred[0][1].round(4) < 0.8:
            print("WARNING, low confidence")
    if pred_class[0] == 2:
        print("Predicted class: (7,14)")
        print("Likelihood:", pred[0][2].round(4))
        if pred[0][2].round(4) < 0.8:
            print("WARNING, low confidence")
    if pred_class[0] == 3:
        print("Predicted class: (15,23)")
        print("Likelihood:", pred[0][3].round(4))
        if pred[0][3].round(4) < 0.8:
            print("WARNING, low confidence")
    if pred_class[0] == 4:
        print("Predicted class: (24,32)")
        print("Likelihood:", pred[0][4].round(4))
        if pred[0][4].round(4) < 0.8:
            print("WARNING, low confidence")
    if pred_class[0] == 5:
        print("Predicted class: (33,43)")
        print("Likelihood:", pred[0][5].round(4))
        if pred[0][5].round(4) < 0.8:
            print("WARNING, low confidence")
    if pred_class[0] == 6:
        print("Predicted class: (44,53)")
        print("Likelihood:", pred[0][6].round(4))
        if pred[0][6].round(4) < 0.8:
            print("WARNING, low confidence")
    if pred_class[0] == 7:
        print("Predicted class: (54,65)")
        print("Likelihood:", pred[0][7].round(4))
        if pred[0][7].round(4) < 0.8:
            print("WARNING, low confidence")
    if pred_class[0] == 8:
        print("Predicted class: (66,100)")
        print("Likelihood:", pred[0][8].round(4))
        if pred[0][8].round(4) < 0.8:
            print("WARNING, low confidence")    
        
choose_image_and_predict()