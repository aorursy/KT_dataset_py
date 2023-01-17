import numpy as np 

import pandas as pd 

import tensorflow as tf #Neural Networks

from sklearn.metrics import roc_auc_score,accuracy_score #metrics for our training

import cv2 

import matplotlib.pyplot as plt

import seaborn as sns # For plotting

from PIL import Image #Image reader python

import os
infected_dir = '../input/cell_images/cell_images/Parasitized/'

noninfected_dir = '../input/cell_images/cell_images/Uninfected/'



img_paths_infected = sorted(

        [

        os.path.join(infected_dir, fname)

        for fname in os.listdir(infected_dir)

        if fname.endswith(".png")

        ])



img_paths_noninfected = sorted(

        [

        os.path.join(noninfected_dir, fname)

        for fname in os.listdir(noninfected_dir)

        if fname.endswith(".png")

        ])

print(f"Number of samples Infected: {len(img_paths_infected)}")

print(f"Number of samples Non-Infected: {len(img_paths_noninfected)}")
image_infected = cv2.imread(img_paths_infected[0])

image_noninfected = cv2.imread(img_paths_noninfected[0])





fig = plt.figure(figsize=(12,6))

gs = fig.add_gridspec(1, 2)

ax1 = fig.add_subplot(gs[0, 0])

ax2 = fig.add_subplot(gs[0, 1])



ax1.imshow(image_noninfected)

ax1.set_title("Non-Infected", size=16)



ax2.imshow(image_infected)

ax2.set_title("Infected", size=16)

plt.show()



print(f"Shape Infected: {image_infected.shape} \nShape Non-Infected: {image_noninfected.shape}")
import random



# Shuffle the data

random.Random(42).shuffle(img_paths_infected)

random.Random(42).shuffle(img_paths_noninfected)



# Choose the percentage of the data that you want for your 

# validation, here I choose 30%.

validation_split = 0.3

val_samples_infected = int(validation_split*len(img_paths_infected))



paths_infected_train = img_paths_infected[val_samples_infected:] 

paths_infected_test = img_paths_infected[:val_samples_infected] 



val_samples_noninfected = int(validation_split*len(img_paths_noninfected))



paths_noninfected_train = img_paths_noninfected[val_samples_noninfected:] 

paths_noninfected_test = img_paths_noninfected[:val_samples_noninfected] 



print(f"Number of training samples for infected / noninfected: {2*len(paths_infected_train)}")

print(f"Number of validation samples for infected / noninfected: {2*len(paths_noninfected_test)}")
train_labels = ['Infected']*len(paths_infected_train)

train_labels.extend(['Noninfected']*len(paths_noninfected_train))



test_labels = ['Infected']*len(paths_infected_test)

test_labels.extend(['Noninfected']*len(paths_noninfected_test))



paths_train = paths_infected_train

paths_train.extend(paths_noninfected_train)



paths_test = paths_infected_test

paths_test.extend(paths_noninfected_test)



print(f"Number of training samples for infected / noninfected: {len(paths_train)}")

print(f"Number of validation samples for infected / noninfected: {len(paths_test)}")
train_df = pd.DataFrame({'train_dir': paths_train})

train_df['train_labels'] =  train_labels



test_df = pd.DataFrame({'test_dir': paths_test})

test_df['test_labels'] = test_labels
from tensorflow.keras.preprocessing.image import ImageDataGenerator



batch_size = 36

img_size = (142, 142)



train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



print("Training dataset:")

train_generator = train_datagen.flow_from_dataframe(

        train_df,

        x_col='train_dir',

        y_col='train_labels',   

        target_size=img_size, 

        batch_size=batch_size,

        class_mode='binary')  



print("Test dataset:")

test_generator = test_datagen.flow_from_dataframe(

        test_df,

        x_col='test_dir',

        y_col='test_labels',   

        target_size= img_size,

        batch_size=batch_size,

        class_mode='binary')  
#seeding random seed so I can get consistent results

tf.set_random_seed(10)

model =  tf.keras.Sequential([

    

    #convolutional layers

    tf.keras.layers.Conv2D(32, (3,3),strides=2 , activation='relu', 

                           input_shape= img_size + (3,), padding='same'),    

    

    tf.keras.layers.Conv2D(64, (3,3), strides=2, activation='relu',padding='same'),        

    

    tf.keras.layers.Conv2D(128, (3,3), strides=2, activation='relu',padding='same'),        

    

    # Dense layers

    tf.keras.layers.Flatten(),    

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



# Compile the model

model.compile(optimizer='adam', 

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()
history = model.fit(train_generator,

                    steps_per_epoch= 400 // batch_size,

                    epochs=10,

                    validation_data=test_generator,

                    validation_steps=50 // batch_size,

                    )
#Plot accuracy and loss per epoch



#accuracy

train_accuracy = history.history['acc']

validation_accuracy = history.history['val_acc']



#loss 

train_loss = history.history['loss']

validation_loss = history.history['val_loss']



epoch_range = range(1,len(train_accuracy)+1)



fig, ax = plt.subplots(1, 2, figsize=(10,5))



#accuracy

ax[0].set_title('Accuracy per Epoch')

sns.lineplot(x=epoch_range,y=train_accuracy,marker='o',ax=ax[0])

sns.lineplot(x=epoch_range,y=validation_accuracy,marker='o',ax=ax[0])

ax[0].legend(['training','validation'])

ax[0].set_xlabel('Epoch')

ax[0].set_ylabel('Accuracy')

ax[0].set_xticks([2,4,6,8,10])

ax[0].set_yticks([.6,.7,.8,.9,1.0])

ax[0].set_yticklabels(['60%','70%','80%','90%','100%'])

#loss

ax[1].set_title('Loss per Epoch')

sns.lineplot(x=epoch_range,y=train_loss,marker='o',ax=ax[1])

sns.lineplot(x=epoch_range,y=validation_loss,marker='o',ax=ax[1])

ax[1].legend(['training','validation'])

ax[1].set_xlabel('Epoch')

ax[1].set_ylabel('Loss')

ax[1].set_xticks([2,4,6,8,10])

plt.show()