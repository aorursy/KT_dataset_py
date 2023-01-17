#Generic Packages

import numpy as np

import os

import pandas as pd

import random

import datetime  # For TensorBoard



#SK Learn

from sklearn.metrics import confusion_matrix

from sklearn.utils import shuffle           



#Plotting Libraries

import seaborn as sn; sn.set(font_scale=1.4)

import matplotlib.pyplot as plt             



#openCV

import cv2                                 



#Tensor Flow

import tensorflow as tf   

from tensorflow.keras.callbacks import Callback

from tensorflow import keras



#Display Progress

from tqdm import tqdm



#Garbage Collector

import gc
class_names = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}



nb_classes = len(class_names)



IMAGE_SIZE = (150, 150)
#Function to Load Images & Labels

def load_data():

    

    datasets = ['../input/image-dataset/_train', '../input/image-dataset/_test']

    output = []

    

    # Iterate through training and test sets

    for dataset in datasets:

        

        images = []

        labels = []

        

        print("Loading {}".format(dataset))

        

        # Iterate through each folder corresponding to a category

        for folder in os.listdir(dataset):

            label = class_names_label[folder]

            

            # Iterate through each image in our folder

            for file in tqdm(os.listdir(os.path.join(dataset, folder))):

                

                # Get the path name of the image

                img_path = os.path.join(os.path.join(dataset, folder), file)

                

                # Open and resize the img

                image = cv2.imread(img_path)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = cv2.resize(image, IMAGE_SIZE) 

                

                # Append the image and its corresponding label to the output

                images.append(image)

                labels.append(label)

                

        images = np.array(images, dtype = 'float32')

        labels = np.array(labels, dtype = 'int32')   

        

        output.append((images, labels))



    return output
#Loading Data (Training & Test Dataset)

(train_images, train_labels), (test_images, test_labels) = load_data()
# Shuffle Training Dataset

train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
#Label Dataset Shape

n_train = train_labels.shape[0]

n_test = test_labels.shape[0]



print ("Number of training examples: {}".format(n_train))

print ("Number of testing examples: {}".format(n_test))

print ("Each image is of size: {}".format(IMAGE_SIZE))
#Scale the data

train_images = train_images / 255.0

test_images = test_images / 255.0
#Build Model



def create_model():

    return tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(8, activation=tf.nn.softmax)

])



#Creating Model

model = create_model()
# Custom Class for LR Finder



class LRFinder(Callback):

    """Callback that exponentially adjusts the learning rate after each training batch between start_lr and

    end_lr for a maximum number of batches: max_step. The loss and learning rate are recorded at each step allowing

    visually finding a good learning rate as per https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html via

    the plot method.

    """



    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10, max_steps: int = 100, smoothing=0.9):

        super(LRFinder, self).__init__()

        self.start_lr, self.end_lr = start_lr, end_lr

        self.max_steps = max_steps

        self.smoothing = smoothing

        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0

        self.lrs, self.losses = [], []



    def on_train_begin(self, logs=None):

        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0

        self.lrs, self.losses = [], []



    def on_train_batch_begin(self, batch, logs=None):

        self.lr = self.exp_annealing(self.step)

        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)



    def on_train_batch_end(self, batch, logs=None):

        logs = logs or {}

        loss = logs.get('loss')

        step = self.step

        if loss:

            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss

            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))

            self.losses.append(smooth_loss)

            self.lrs.append(self.lr)



            if step == 0 or loss < self.best_loss:

                self.best_loss = loss



            if smooth_loss > 4 * self.best_loss or tf.math.is_nan(smooth_loss):

                self.model.stop_training = True



        if step == self.max_steps:

            self.model.stop_training = True



        self.step += 1



    def exp_annealing(self, step):

        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)



    def plot(self):

        fig, ax = plt.subplots(1, 1)

        ax.set_ylabel('Loss')

        ax.set_xlabel('Learning Rate')

        ax.set_xscale('log')

        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

        ax.plot(self.lrs, self.losses)
# -- Model Parameters --



epochs = 10

batch_size = 64

val_size = 0.1
# -- Implementing Technique#1 --

lr_finder = LRFinder()



# Optimizer & Learning Rate

optimizer = keras.optimizers.Adam()



#Compile Model

model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])



# Fit Model

history = model.fit(train_images, train_labels, 

                    batch_size=batch_size, 

                    epochs=epochs, 

                    validation_split = val_size, 

                    verbose=-1,

                    callbacks=[lr_finder]             # LR Finder

                   )





# Plot Loss vs Learning Rate

lr_finder.plot()
# UPDATED Learning Rate

lr = 1e-02

optimizer = keras.optimizers.Adam(learning_rate = lr)





#Re-Compile Model

model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])





# Re-Fit the Model

history = model.fit(train_images, train_labels, 

                    batch_size=batch_size, 

                    epochs=epochs, 

                    validation_split = val_size, 

                    verbose=-1)





# Evaluate Model

test_loss = model.evaluate(test_images, test_labels, verbose=0)

print("\n Model Accuracy (after): ",'{:.2%}'.format(test_loss[1]))
#garbage collection to save memory

gc.collect()
# -- Define Logging Directory & TimeStamp --

#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d%m%Y-%H%M")



# --TensorBoard CallBack --

#tensorBoard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)



# -- Fitting the Model with TensorBoard CallBack --

#history = model2.fit(train_images, train_labels, 

#                    batch_size=batch_size, 

#                    epochs=epochs, 

#                    validation_split = val_size, 

#                    callbacks=[tensorBoard_callback],

#                    verbose=-1)
# -- Load the TensorBoard notebook extension --

#%load_ext tensorboard



# -- Starting TensorBoard --

#%tensorboard --logdir logs/fit