import os

from keras.models import Sequential # Initialise our neural network model as a sequential network

from keras.layers import Conv2D # Convolution operation

from keras.layers import MaxPooling2D # Maxpooling function

from keras.layers import Flatten # Converting 2D arrays into a 1D linear vector.

from keras.layers import Dense # Perform the full connection of the neural network

from keras.layers import Dropout # Perform Dropout on layers

from keras.layers import BatchNormalization

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator

from keras.backend.tensorflow_backend import clear_session

from keras.optimizers import SGD, Adam

from IPython.display import display

from PIL import Image

import cv2

import numpy as np

from shutil import copyfile

import tensorflow as tf

from sklearn.metrics import accuracy_score

from skimage import io, transform

from tensorflow import set_random_seed

from keras.metrics import binary_accuracy

from keras.regularizers import l2,l1

from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

print(os.listdir('../input/train'))
set_random_seed(2)

train_path = '../input/train/'

im = 300

channels = 3

batch_size = 32

epochs = 15
dataset_home = '../input/'

print(os.listdir(dataset_home + 'test/'))
def reshaped_image(image):

    return transform.resize(image,(im, im, 3)) # (cols (width), rows (height)) and don't use np.resize()



def load_images_from_folder():

    Images = os.listdir(train_path)

    train_images = []

    train_labels = []

    i=0

    for image in Images:

            l = [] # [cat,dog]

            i +=1

            if image.find('cat') != -1:

                path = os.path.join(train_path, image)

                img = cv2.imread(path)

                train_images.append(reshaped_image(img))

                l = 0 

                train_labels.append(l)

            if image.find('dog') != -1:

                path = os.path.join(train_path, image)

                img = cv2.imread(path)

                train_images.append(reshaped_image(img))

                l = 1 

                train_labels.append(l)

            print("{} th Image found".format(i))

    return np.array(train_images), np.array(train_labels)

        

def train_test_split(data, labels, fraction):

    # standardized Inputs

    for x in range(len(data)):

        for i in range(channels):

            min = np.min(data[x,:,:,i])

            max = np.max(data[x,:,:,i])

            data[x,:,:,i] = data[x,:,:,i] - min

            data[x,:,:,i] = data[x,:,:,i]/max

    print("Done!")

    index = int(len(data)*fraction)

    idx = np.random.randint(0,len(data),len(data))

    idx_train = idx[:index]

    idx_val = idx[index:]

    return data[idx_train], labels[idx_train], data[idx_val], labels[idx_val]
valid_datagen = ImageDataGenerator(rescale = 1.0/255.0)

train_datagen = ImageDataGenerator(rescale = 1.0/255.0,

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True)
train_it = train_datagen.flow_from_directory(dataset_home + 'train/', class_mode = 'binary', batch_size = 32, target_size = (im,im))

test_it = valid_datagen.flow_from_directory(dataset_home + 'test/', class_mode = 'binary',batch_size = 32, target_size = (im,im))
def model():

    cnn = Sequential()

    

    cnn.add(Conv2D(32, (5,5), input_shape = (im, im, 3), padding='same', activation = 'relu'))

    cnn.add(BatchNormalization())

    cnn.add(Conv2D(32,(5,5),padding = 'same',activation = 'relu'))

    cnn.add(BatchNormalization())

    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    cnn.add(Dropout(0.1))

    

    cnn.add(Conv2D(64,(5,5),padding = 'same',activation = 'relu'))

    cnn.add(BatchNormalization())

    cnn.add(Conv2D(64,(5,5),padding = 'same',activation = 'relu'))

    cnn.add(BatchNormalization())

    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    cnn.add(Dropout(0.2))

    

    cnn.add(Conv2D(128,(5,5),padding = 'same',activation = 'relu'))

    cnn.add(BatchNormalization())

    cnn.add(Conv2D(128,(5,5),padding = 'same',activation = 'relu'))

    cnn.add(BatchNormalization())

    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    cnn.add(Dropout(0.2))

    

    cnn.add(Conv2D(256,(5,5),padding = 'same',activation = 'relu'))

    cnn.add(BatchNormalization())

    cnn.add(Conv2D(256,(5,5),padding = 'same',activation = 'relu'))

    cnn.add(BatchNormalization())

    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    cnn.add(Dropout(0.2))

    

    cnn.add(Conv2D(512,(5,5),padding = 'same',activation = 'relu'))

    cnn.add(BatchNormalization())

    cnn.add(Conv2D(512,(5,5),padding = 'same',activation = 'relu'))

    cnn.add(BatchNormalization())

    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    cnn.add(Dropout(0.2))

 

    cnn.add(Flatten())

    cnn.add(Dense(256, activation = 'relu'))

    cnn.add(BatchNormalization())

    cnn.add(Dropout(0.2))

    cnn.add(Dense(256, activation = 'relu'))

    cnn.add(BatchNormalization())

    cnn.add(Dropout(0.2))

    cnn.add(Dense(1, activation = 'sigmoid'))

    opt = SGD(lr = 0.001, momentum = 0.9, nesterov = True)

    cnn.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'] )

    print(cnn.summary())

    return cnn
def summarize_diagnostics(history):

	# plot loss

	plt.subplot(211)

	plt.title('Cross Entropy Loss')

	plt.plot(history.history['loss'], color='blue', label='train')

	plt.plot(history.history['val_loss'], color='orange', label='test')

	# plot accuracy

	plt.subplot(212)

	plt.title('Classification Accuracy')

	plt.plot(history.history['acc'], color='blue', label='train')

	plt.plot(history.history['val_acc'], color='orange', label='test')

	# save plot to file
model = model()
#es = EarlyStopping(monitor='val_loss', min_delta=0.001, verbose=1, mode='auto',restore_best_weights = True) 
from keras.callbacks import Callback

import keras.backend as K

import numpy as np



class SGDRScheduler(Callback):

    '''Cosine annealing learning rate scheduler with periodic restarts.

    # Usage

        ```python

            schedule = SGDRScheduler(min_lr=1e-5,

                                     max_lr=1e-2,

                                     steps_per_epoch=np.ceil(epoch_size/batch_size),

                                     lr_decay=0.9,

                                     cycle_length=5,

                                     mult_factor=1.5)

            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])

        ```

    # Arguments

        min_lr: The lower bound of the learning rate range for the experiment.

        max_lr: The upper bound of the learning rate range for the experiment.

        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 

        lr_decay: Reduce the max_lr after the completion of each cycle.

                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.

        cycle_length: Initial number of epochs in a cycle.

        mult_factor: Scale epochs_to_restart after each full cycle completion.

    # References

        Blog post: jeremyjordan.me/nn-learning-rate

        Original paper: http://arxiv.org/abs/1608.03983

    '''

    def __init__(self,

                 min_lr,

                 max_lr,

                 steps_per_epoch,

                 lr_decay=1,

                 cycle_length=10,

                 mult_factor=2):



        self.min_lr = min_lr

        self.max_lr = max_lr

        self.lr_decay = lr_decay



        self.batch_since_restart = 0

        self.next_restart = cycle_length



        self.steps_per_epoch = steps_per_epoch



        self.cycle_length = cycle_length

        self.mult_factor = mult_factor



        self.history = {}



    def clr(self):

        '''Calculate the learning rate.'''

        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)

        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))

        return lr



    def on_train_begin(self, logs={}):

        '''Initialize the learning rate to the minimum value at the start of training.'''

        logs = logs or {}

        K.set_value(self.model.optimizer.lr, self.max_lr)



    def on_batch_end(self, batch, logs={}):

        '''Record previous batch statistics and update the learning rate.'''

        logs = logs or {}

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)



        self.batch_since_restart += 1

        K.set_value(self.model.optimizer.lr, self.clr())



    def on_epoch_end(self, epoch, logs={}):

        '''Check for end of current cycle, apply restarts when necessary.'''

        if epoch + 1 == self.next_restart:

            self.batch_since_restart = 0

            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)

            self.next_restart += self.cycle_length

            self.max_lr *= self.lr_decay

            self.best_weights = self.model.get_weights()



    def on_train_end(self, logs={}):

        '''Set weights to the values from the end of the most recent cycle for best performance.'''

        self.model.set_weights(self.best_weights)
schedule = SGDRScheduler(min_lr=2e-6,

                                     max_lr=1e-2,

                                     steps_per_epoch=len(train_it),

                                     lr_decay=0.95,

                                     cycle_length=5,

                                     mult_factor=1.5)
history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data = test_it, 

                              validation_steps = len(test_it), epochs= 1 ,verbose = 1)

                              #callbacks=[schedule]) 
_,acc = model.evaluate_generator(test_it, steps = len(test_it), verbose=1)
print('> %.3f' %(acc*100))

summarize_diagnostics(history)