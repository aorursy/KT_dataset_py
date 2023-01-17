!pip install tensorflow-gpu==2.0.0-beta1
import tensorflow as tf

import pandas as pd

import numpy as np



print(tf.__version__)

print(tf.test.is_gpu_available())
# load data from csv file

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
from sklearn.model_selection import train_test_split

from keras.utils import np_utils



batch_size = 128

num_classes = 10

smooth_factor = 0.1

nets = 3



def smooth_labels(y, smooth_factor):

    y *= 1 - smooth_factor

    y += smooth_factor / y.shape[1]

    return y



# label one_hot encoding and smoothing

def label_process(y, num_classes):

    y = np_utils.to_categorical(y, num_classes)

    y = smooth_labels(y, smooth_factor)

    return y



# Split train and valid dataset

y_train_full = train['label'].astype(np.float32)

X_train_full = train.drop('label', axis=1)

X_train_full = X_train_full.values.reshape((-1, 28, 28, 1))

X_train_full = X_train_full.astype(np.float32) / 255.



# test dataset

X_test = test.values.reshape((-1, 28, 28, 1))

X_test = X_test.astype(np.float32) / 255.

print(X_test.shape)



# free space

del train, test
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D

from tensorflow.keras.layers import BatchNormalization, Activation, Dropout

from tensorflow.keras.regularizers import l2

from tensorflow.python.keras.optimizers import RMSprop



# l2 regularizer

weight_decay = 1e-6



def build_model():

    model = Sequential()

    # modified the filter size: (5,5) -> (3,3) * 2

    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal', input_shape=(28, 28, 1)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(160, (1, 1),padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    

    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),kernel_initializer='he_normal'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),kernel_initializer='he_normal'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

    model.add(Dropout(0.2))

    

    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=l2(weight_decay),kernel_initializer='he_normal'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(num_classes, (1, 1), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal'))

    model.add(BatchNormalization())

    model.add(Activation('relu')) 

    

    model.add(GlobalAveragePooling2D())

    model.add(Activation('softmax'))

    

    optimizer = RMSprop(lr=1e-3, rho=0.9, epsilon=1e-08, decay= 0)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
models = []

for i in range(nets):

    models.append(build_model())
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping



learning_rate_reduction = ReduceLROnPlateau(

    monitor='val_loss',

    patience=2,

    verbose=2,

    factor=0.4,

    min_delta=1e-4)

early_stopping = EarlyStopping(

    monitor='val_loss',

    min_delta=1e-5,

    patience=5,

    verbose=2,

    mode='min',

    restore_best_weights=True)



aug_factor = 12 

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range= aug_factor,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = aug_factor / 100, # Randomly zoom image 

        width_shift_range= aug_factor / 100,  # randomly shift images horizontally (fraction of total width)

        height_shift_range= aug_factor / 100,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images horizontally

        vertical_flip=False)  # randomly flip images vertically



histories = []

for i in range(nets):    

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, 

                                                      test_size=0.25, stratify=y_train_full)

    y_train = label_process(y_train, num_classes)

    y_valid = label_process(y_valid, num_classes)

    

    datagen.fit(X_train)

    history = models[i].fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                                  steps_per_epoch=X_train.shape[0]//batch_size,

                                  epochs=50,

                                  callbacks=[learning_rate_reduction, early_stopping],

                                  validation_data=(X_valid,y_valid),

                                  verbose=1)

    histories.append(history)

    print('Network in Network {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}'.

          format(i+1,max(history.epoch),max(history.history['accuracy']),max(history.history['val_accuracy'])))
import matplotlib.pyplot as plt

plt.figure(figsize=(12,12))

plt.subplot(2,1,1)

for i in range(nets):

    history=histories[i]

    epoch_range = 1 + np.arange(len(history.history['accuracy']))

    plt.plot(epoch_range,history.history['loss'],'g-',label='Training loss')

    plt.plot(epoch_range,history.history['val_loss'],'r--',label='Validation loss')

    plt.legend(loc='best',shadow=True)

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    # plt.xlim([0,epochs])

    plt.ylim([0,0.2])





plt.subplot(2,1,2)

for i in range(nets):

    history=histories[i]

    epoch_range = 1 + np.arange(len(history.history['accuracy']))

    plt.plot(epoch_range,history.history['accuracy'],'g-',label='Training accuracy')

    plt.plot(epoch_range,history.history['val_accuracy'],'r--',label='Validation accuracy')

    plt.legend(loc='best',shadow=True)

    plt.xlabel('Epoch')

    plt.ylabel('Accuracy')

    # plt.xlim([0,epochs])

    plt.ylim([0.95,1])

plt.show()
# import matplotlib.pyplot as plt



# def plot_history(history):

#     hist = pd.DataFrame(history.history)

#     hist['epoch'] = history.epoch



#     plt.figure()

#     plt.xlabel('Epoch')

#     plt.ylabel('CategoricalCrossentropy')

#     plt.plot(hist['epoch'], hist['accuracy'],

#              label='Train Accuracy')

#     plt.plot(hist['epoch'], hist['val_accuracy'],

#              label = 'Val Accuracy')

#     plt.ylim([0.98, 1.01])

#     plt.legend()



# plot_history(history)
results = np.zeros((len(X_test),10),dtype='float')

for i in range(nets):

    model = models[i]

    results += model.predict(X_test, batch_size=128,verbose=1)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
# import numpy as np

# # predict results

# results = model.predict(X_test)



# # select the indix with the maximum probability

# results = np.argmax(results,axis = 1)

# results = pd.Series(results,name="Label")



# submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

# submission.to_csv("submission.csv",index=False)