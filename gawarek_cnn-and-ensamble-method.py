# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import keras

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

test = pd.read_csv("../input/test.csv")

data = data.sample(frac=1).reset_index(drop=True)

Y_train = data['label']

X_train = data.drop(labels = ["label"],axis = 1) 



X_train = X_train.values.reshape(-1, 28, 28, 1)

X_test = test.values.reshape(-1, 28, 28, 1)





X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size = 0.1)



X_test = X_test / 255.0

X_train = X_train / 255.0

X_dev = X_dev / 255.0



Y_dev = keras.utils.to_categorical(Y_dev, num_classes = 10)

Y_train = keras.utils.to_categorical(Y_train, num_classes = 10)



del data

del test
train_gen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



train_gen.fit(X_train)
def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize=(20,10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
def leNet5(X_in, filters=(32, 64), neurons=(256, 128)):

    

    F1, F2 = filters #6, 16

    N1, N2 = neurons #120, 84

    

    X = keras.layers.Conv2D(filters=F2, kernel_size=(5,5), strides=(1,1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.05))(X_in)

    X = keras.layers.BatchNormalization(axis=3)(X)

    X = keras.layers.Activation('relu')(X)

    

    X = keras.layers.MaxPool2D(pool_size=(2,2))(X)

    

    X = keras.layers.Conv2D(filters=F1, kernel_size=(5,5), strides=(1,1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.03))(X)

    X = keras.layers.BatchNormalization(axis=3)(X)

    X = keras.layers.Activation('relu')(X)

    

    X = keras.layers.MaxPool2D(pool_size=(2,2))(X)

    

    X = keras.layers.Flatten()(X)

    

    X = keras.layers.Dense(N1, kernel_regularizer=keras.regularizers.l2(0.02))(X)

    X = keras.layers.BatchNormalization(axis=1)(X)

    X = keras.layers.Activation('relu')(X)

    

    X = keras.layers.Dense(N2, kernel_regularizer=keras.regularizers.l2(0.01))(X)

    X = keras.layers.BatchNormalization(axis=1)(X)

    X = keras.layers.Activation('relu')(X)

    

    model = keras.models.Model(inputs = X_in, outputs = X, name='LeNet-5-part')

    

    return model
X_in = keras.layers.Input(X_train[0].shape)



model_part = leNet5(X_in)



X = keras.layers.Dense(10, activation='softmax')(model_part.outputs[0])



model_full = keras.models.Model(inputs = X_in, outputs = X, name='LeNet-5')



model_full.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=2, min_lr=0.00001)

history1 = model_full.fit_generator(train_gen.flow(X_train,Y_train, batch_size=128),

                                epochs = 30, validation_data = (X_dev,Y_dev), 

                                steps_per_epoch=512, callbacks=[reduce_lr])
plt.figure(figsize=(20,10))

plt.plot(history1.history['acc'])

plt.plot(history1.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
Y_pred = model_full.predict(X_dev)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_dev,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
def inception(X_in, filters=(32, 32), neurons=(256, 128)):

    

    def create_branch(X, fs, size, strides):

        X = keras.layers.Conv2D(filters=fs, kernel_size=size, strides=strides, padding='same', kernel_regularizer=keras.regularizers.l2(0.05))(X)

        X = keras.layers.BatchNormalization(axis=3)(X)

        X = keras.layers.Activation('relu')(X)

        return X

    

    def create_layer(X, filters_1, filters_2):

        X = keras.layers.concatenate([

            create_branch(X, filters_1, (3,3), (1,1)),

            create_branch(X, filters_1, (5,5), (1,1)),

            create_branch(X, filters_1, (7,7), (1,1))

        ])

        X = keras.layers.Conv2D(filters=filters_2, kernel_size=(1,1), strides=(1,1), padding='same', kernel_regularizer=keras.regularizers.l2(0.03))(X)

        X = keras.layers.BatchNormalization(axis=3)(X)

        X = keras.layers.Activation('relu')(X)

        return X

    

    F1, F2 = filters #8, 16

    N1, N2 = neurons #120, 84

    

    X = create_layer(X_in, F1, F2)

    X = create_layer(X, F1, F2)

    X = create_layer(X, F1, F2)

    X = keras.layers.MaxPool2D(pool_size=(2,2))(X)

    

    X = create_layer(X, F1, F2)

    X = keras.layers.MaxPool2D(pool_size=(2,2))(X)

    X = keras.layers.Conv2D(filters=1024, kernel_size=(7,7), strides=(1,1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(X)

    X = keras.layers.Flatten()(X)

    

    X = keras.layers.Dense(N1, kernel_regularizer=keras.regularizers.l2(0.02))(X)

    X = keras.layers.BatchNormalization(axis=1)(X)

    X = keras.layers.Activation('relu')(X)

    

    X = keras.layers.Dense(N2, kernel_regularizer=keras.regularizers.l2(0.01))(X)

    X = keras.layers.BatchNormalization(axis=1)(X)

    X = keras.layers.Activation('relu')(X)

    

    model = keras.models.Model(inputs = X_in, outputs = X, name='Inception-part')

    

    return model
X_in = keras.layers.Input(X_train[0].shape)



model_2_part = inception(X_in)



X = keras.layers.Dense(10, activation='softmax')(model_2_part.outputs[0])



model_2_full = keras.models.Model(inputs = X_in, outputs = X, name='Inception')



model_2_full.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model_2_full.fit_generator(train_gen.flow(X_train,Y_train, batch_size=128),

                                epochs = 30, validation_data = (X_dev,Y_dev), 

                                steps_per_epoch=512, callbacks=[reduce_lr])
plt.figure(figsize=(20,10))

plt.plot(history2.history['acc'])

plt.plot(history2.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
Y_pred = model_2_full.predict(X_dev)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_dev,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
def custom(X_in, filters=(32, 64), neurons=(256, 128)):

    

    F1, F2 = filters #6, 16

    N1, N2 = neurons #120, 84

    

    X = keras.layers.Conv2D(filters=F1, kernel_size=(5,5), strides=(1,1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(X_in)

    X = keras.layers.BatchNormalization(axis=3)(X)

    X = keras.layers.Activation('relu')(X)

    

    X = keras.layers.Conv2D(filters=F2, kernel_size=(5,5), strides=(1,1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(X)

    X = keras.layers.BatchNormalization(axis=3)(X)

    X = keras.layers.Activation('relu')(X)

    

    X = keras.layers.Conv2D(filters=F1, kernel_size=(5,5), strides=(1,1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(X_in)

    X = keras.layers.BatchNormalization(axis=3)(X)

    X = keras.layers.Activation('relu')(X)

    

    X = keras.layers.Conv2D(filters=F2, kernel_size=(3,3), strides=(1,1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(X)

    X = keras.layers.BatchNormalization(axis=3)(X)

    X = keras.layers.Activation('relu')(X)

    

    X = keras.layers.MaxPool2D(pool_size=(2,2), padding='same')(X)

    

    X = keras.layers.Conv2D(filters=1024, kernel_size=(7,7), strides=(1,1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(X)

    X = keras.layers.BatchNormalization(axis=3)(X)

    X = keras.layers.Activation('relu')(X)

    

    X = keras.layers.Flatten()(X)

    

    X = keras.layers.Dense(N1, kernel_regularizer=keras.regularizers.l2(0.01))(X)

    X = keras.layers.BatchNormalization(axis=1)(X)

    X = keras.layers.Activation('relu')(X)

    

    X = keras.layers.Dense(N2, kernel_regularizer=keras.regularizers.l2(0.01))(X)

    X = keras.layers.BatchNormalization(axis=1)(X)

    X = keras.layers.Activation('relu')(X)

    

    model = keras.models.Model(inputs = X_in, outputs = X, name='LeNet-5-part')

    

    return model
X_in = keras.layers.Input(X_train[0].shape)



model_3_part = custom(X_in)



X = keras.layers.Dense(10, activation='softmax')(model_3_part.outputs[0])



model_3_full = keras.models.Model(inputs = X_in, outputs = X, name='Custom')



model_3_full.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history3 = model_3_full.fit_generator(train_gen.flow(X_train,Y_train, batch_size=128),

                                epochs = 30, validation_data = (X_dev,Y_dev), 

                                steps_per_epoch=512, callbacks=[reduce_lr])
Y_pred = model_3_full.predict(X_dev)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_dev,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
plt.figure(figsize=(20,10))

plt.plot(history3.history['acc'])

plt.plot(history3.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
Y_pred = (model_2_full.predict(X_dev) + model_full.predict(X_dev) + model_3_full.predict(X_dev)) / 3

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_dev,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
def submit_results(model, file_name, test): 

    results = model.predict(test)

    results = np.argmax(results, axis = 1)

    results = pd.Series(results, name="Label")

    submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"),results],axis = 1)

    submission.to_csv(file_name, index=False)

    

def ensamble_submission(model1, model2, model3, test): 

    results = (model1.predict(test) + model2.predict(test) + model3.predict(test)) / 2

    results = np.argmax(results, axis = 1)

    results = pd.Series(results, name="Label")

    submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"),results],axis = 1)

    submission.to_csv("ensamble_submission.csv", index=False)

    

submit_results(model_full, "lenet-5-submission.csv", X_test)

submit_results(model_2_full, "inception-submission.csv", X_test)

ensamble_submission(model_full, model_2_full, model_3_full, X_test)