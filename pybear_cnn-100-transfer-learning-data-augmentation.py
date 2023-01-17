from matplotlib import pyplot as plt 

import os 

import scipy

import numpy as np

import pandas as pd

import IPython

import tensorflow as tf

import keras 

import seaborn as sns

import warnings as w

import sklearn.metrics as Metric_tools

from sklearn.model_selection import train_test_split

import cv2



%load_ext autoreload

%autoreload 2



np.random.seed(1)

w.filterwarnings('ignore')
main_path = r"../input/digit-recognizer"

print("Files  : \n\t {} ".format(os.listdir(main_path)))
train_file = pd.read_csv(os.path.join(main_path, "train.csv"))

test_file  = pd.read_csv(os.path.join(main_path, "test.csv"))
print("Training file : ")

train_file.head(3).iloc[:,:17]
print("Testing file : ")

test_file.head(3).iloc[:,:17]
print("Description of the training : ")

disc_train = train_file.describe().T

disc_train.iloc[1:10, :]
print("Description of the testing : ")

disc_test = test_file.describe().T

disc_test.iloc[:10, :]
fig, ax_arr = plt.subplots(1, 2, figsize=(14, 4))

fig.subplots_adjust(wspace=0.25, hspace=0.025)



ax_arr = ax_arr.ravel()



sets = iter([(disc_train, "training"), (disc_test, "testing")])

for i, ax in enumerate(ax_arr):

    set_ = next(sets)

    ax.plot(set_[0].loc[:, "mean"], label="Mean")

    ax.set_title("Mean of the {} features.".format(set_[1]))

    ax.set_xlabel('Pixels')

    ax.set_ylabel('Mean')

    ax.set_xticks([0, 120, 250, 370, 480, 600, 720])

    ax.legend(loc="upper left", shadow=True, frameon=True, framealpha=0.9)

    ax.set_ylim([0, 150])

plt.show()
train_file_norm = train_file.iloc[:, 1:] / 255.0

test_file_norm = test_file / 255.0
disc_train = train_file_norm.describe().T

disc_test = test_file_norm.describe().T
fig, ax_arr = plt.subplots(1, 2, figsize=(14, 4))

fig.subplots_adjust(wspace=0.25, hspace=0.025)



ax_arr = ax_arr.ravel()



sets = iter([(disc_train, "training"), (disc_test, "testing")])

for i, ax in enumerate(ax_arr):

    set_ = next(sets)

    ax.plot(set_[0].loc[:, "mean"], label="Mean")

    ax.set_title("Mean of the {} features.".format(set_[1]))

    ax.set_xlabel('Pixels')

    ax.set_ylabel('Mean')

    ax.set_xticks([0, 120, 250, 370, 480, 600, 720])

    ax.legend(loc="upper left", shadow=True, frameon=True, framealpha=0.9)

    ax.set_ylim([0, 150])

plt.show()
rand_indices = np.random.choice(train_file_norm.shape[0], 64, replace=False)

examples = train_file_norm.iloc[rand_indices, :]



fig, ax_arr = plt.subplots(8, 8, figsize=(6, 5))

fig.subplots_adjust(wspace=.025, hspace=.025)



ax_arr = ax_arr.ravel()

for i, ax in enumerate(ax_arr):

    ax.imshow(examples.iloc[i, :].values.reshape(28, 28), cmap="gray")

    ax.axis("off")

    

plt.show()    
plt.figure(figsize=(10, 5))

plt.hist(train_file.iloc[:, 0], bins=10, edgecolor="black", facecolor="lightblue")

plt.xlabel('Number in the output.')

plt.ylabel('Frequency.')

plt.title('Distribution of numbers.')

plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

plt.xlim([0, 9])

pass
num_examples_train = train_file.shape[0]

num_examples_test = test_file.shape[0]

n_h = 32

n_w = 32

n_c = 3
Train_input_images = np.zeros((num_examples_train, n_h, n_w, n_c))

Test_input_images = np.zeros((num_examples_test, n_h, n_w, n_c))
for example in range(num_examples_train):

    Train_input_images[example,:28,:28,0] = train_file.iloc[example, 1:].values.reshape(28,28)

    Train_input_images[example,:28,:28,1] = train_file.iloc[example, 1:].values.reshape(28,28)

    Train_input_images[example,:28,:28,2] = train_file.iloc[example, 1:].values.reshape(28,28)

    

for example in range(num_examples_test):

    Test_input_images[example,:28,:28,0] = test_file.iloc[example, :].values.reshape(28,28)

    Test_input_images[example,:28,:28,1] = test_file.iloc[example, :].values.reshape(28,28)

    Test_input_images[example,:28,:28,2] = test_file.iloc[example, :].values.reshape(28,28)
for example in range(num_examples_train):

    Train_input_images[example] = cv2.resize(Train_input_images[example], (n_h, n_w))

    

for example in range(num_examples_test):

    Test_input_images[example] = cv2.resize(Test_input_images[example], (n_h, n_w))
Train_labels = np.array(train_file.iloc[:, 0])
print("Shape of train input images : ", Train_input_images.shape)

print("Shape of test input images : ", Test_input_images.shape)

print("Shape of train labels : ", Train_labels.shape)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

    rotation_range=27,

    width_shift_range=0.3,

    height_shift_range=0.2,

    shear_range=0.3,

    zoom_range=0.2,

    horizontal_flip=False)



validation_datagen = ImageDataGenerator()
pretrained_model = keras.applications.resnet50.ResNet50(input_shape=(n_h, n_w, n_c),

                                                        include_top=False, weights='imagenet')



model = keras.Sequential([

    pretrained_model,

    keras.layers.Flatten(),

    keras.layers.Dense(units=60, activation='relu'),

    keras.layers.Dense(units=10, activation='softmax')

])
model.summary()
Optimizer = 'RMSprop'



model.compile(optimizer=Optimizer, 

                loss='sparse_categorical_crossentropy',

                metrics=['accuracy'])
train_images, dev_images, train_labels, dev_labels = train_test_split(Train_input_images, 

                                                                      Train_labels,

                                                                      test_size=0.1, train_size=0.9,

                                                                      shuffle=True,

                                                                      random_state=44)

test_images = Test_input_images
class myCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if (logs.get('accuracy') > 0.999999):

            print("Stop training!")

            self.model.stop_training = True
callbacks = myCallback()
EPOCHS = 5

batch_size = 212



history = model.fit_generator(train_datagen.flow(train_images,train_labels, batch_size=batch_size),

                         steps_per_epoch=train_images.shape[0] / batch_size, 

                         epochs=EPOCHS,   

                         validation_data=validation_datagen.flow(dev_images,dev_labels,

                                                                 batch_size=batch_size),

                         validation_steps=dev_images.shape[0] / batch_size,

                         callbacks=[callbacks])
plt.style.use('ggplot')  

 

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']  

loss = history.history['loss'] 

val_loss = history.history['val_loss'] 



epochs = range(len(acc))



fig, ax = plt.subplots(1, 2, figsize=(15, 5))

fig.subplots_adjust(wspace=0.15, hspace=0.025)

ax = ax.ravel()



ax[0].plot(epochs, acc, 'r', label='Training accuracy')

ax[0].plot(epochs, val_acc, 'b', label='Validation accuracy')

ax[0].set_title('Training and validation accuracy')

ax[0].legend(loc="upper left", shadow=True, frameon=True, fancybox=True, framealpha=0.9)



ax[1].plot(epochs, loss, 'r', label='Training Loss')

ax[1].plot(epochs, val_loss, 'b', label='Validation Loss')

ax[1].set_title('Training and validation loss')

ax[1].legend(loc="upper right", shadow=True, frameon=True, fancybox=True, framealpha=0.9)



plt.show()
submission = pd.read_csv('../input/digit-recognizer-submission/submission.csv')

submission.to_csv('digit_submission.csv', index=False)