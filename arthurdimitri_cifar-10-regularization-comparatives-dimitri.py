# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from keras.datasets import cifar10

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization

from keras import regularizers

from keras.optimizers import RMSprop, Adadelta, Adam, Adamax, Nadam

from keras.callbacks import Callback

from keras import backend as K

from IPython.display import Image

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import numpy as np

from keras.utils import np_utils
(X_train, y_train),(X_validation, y_validation) = cifar10.load_data()
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

plt.figure(figsize=(15,10))

sns.set_style('white')

for i in range(50):  

    plt.subplot(5, 10, i+1)

    plt.imshow(X_train[i].reshape((32, 32, 3)),cmap=plt.cm.hsv)

    plt.title(labels[y_train.reshape(1,len(y_train))[0][i]])

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.show()
X_train = X_train.astype('float32')

X_validation = X_validation.astype('float32')

std_x_valid = np.std(X_validation)

mean_x_train = np.mean(X_train)

mean_x_valid = np.mean(X_validation)

X_train = X_train/mean_x_train

X_validation = X_validation/mean_x_valid

y_temp = y_validation
y_train = np_utils.to_categorical(y_train.transpose()).reshape(50000,10)

y_validation = np_utils.to_categorical(y_validation.transpose()).reshape(10000,10)

input_shape = (32,32,3)
y_train = y_train 
def CnnModel():

    

    model = Sequential()

    model.add(Conv2D(32, (3, 3),strides=(1,1), padding='same', kernel_regularizer=regularizers.l1_l2(1e-4,1e-4),input_shape=input_shape))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3),strides=(1,1), kernel_regularizer=regularizers.l1_l2(1e-4,1e-4)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Conv2D(64, (3, 3),strides=(1,1),padding='same', kernel_regularizer=regularizers.l1_l2(1e-4,1e-4)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3),strides=(1,1), kernel_regularizer=regularizers.l1_l2(1e-4,1e-4)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1)))

    

    model.add(Flatten())

    model.add(Dense(512))

    model.add(Activation('relu'))

    model.add(BatchNormalization())



    model.add(Dense(10))

    model.add(Activation('softmax'))

    

    return model
def CnnModel():

    

    model = Sequential()

    model.add(Conv2D(32, (3, 3),strides=(1,1), padding='same', kernel_regularizer=regularizers.l1_l2(1e-4,1e-4),input_shape=input_shape))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3),strides=(1,1), kernel_regularizer=regularizers.l1_l2(1e-4,1e-4)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(64, (3, 3),strides=(1,1),padding='same', kernel_regularizer=regularizers.l1_l2(1e-4,1e-4)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3),strides=(1,1), kernel_regularizer=regularizers.l1_l2(1e-4,1e-4)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1)))

    model.add(Dropout(0.3))

    

    model.add(Conv2D(128, (3, 3),strides=(1,1),padding='same', kernel_regularizer=regularizers.l1_l2(1e-4,1e-4)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3),strides=(1,1),kernel_regularizer=regularizers.l1_l2(1e-4,1e-4)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1)))

    model.add(Dropout(0.35))

    

    model.add(Flatten())

    model.add(Dense(512))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))



    model.add(Dense(10))

    model.add(Activation('softmax'))

    

    return model
def CnnModel():

    

    model = Sequential()

    model.add(Conv2D(32, (3, 3),strides=(1,1), padding='same', kernel_regularizer=regularizers.l1_l2(1e-4,1e-4),input_shape=input_shape))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3),strides=(1,1), kernel_regularizer=regularizers.l1_l2(1e-4,1e-4)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(64, (3, 3),strides=(1,1),padding='same', kernel_regularizer=regularizers.l1_l2(1e-4,1e-4)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3),strides=(1,1), kernel_regularizer=regularizers.l1_l2(1e-4,1e-4)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1)))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(128, (3, 3),strides=(1,1),padding='same', kernel_regularizer=regularizers.l1_l2(1e-4,1e-4)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3),strides=(1,1),kernel_regularizer=regularizers.l1_l2(1e-4,1e-4)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1)))

    model.add(Dropout(0.3))

    

    model.add(Conv2D(256, (3, 3),strides=(1,1),padding='same', kernel_regularizer=regularizers.l1_l2(1e-4,1e-4)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3),strides=(1,1),kernel_regularizer=regularizers.l1_l2(1e-4,1e-4)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1)))

    model.add(Dropout(0.4))

    

    model.add(Flatten())

    model.add(Dense(512))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4

                     ))



    model.add(Dense(10))

    model.add(Activation('softmax'))

    

    return model
'''K.clear_session()

#history = model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1,validation_data=(X_validation, y_validation),shuffle=True)



import tensorflow as tf



class MyThresholdCallback(tf.keras.callbacks.Callback):

    def __init__(self, threshold):

        super(MyThresholdCallback, self).__init__()

        self.threshold = threshold



    def on_epoch_end(self, epoch, logs=None): 

        val_acc = logs["val_accuracy"]

        if val_acc >= self.threshold:

            self.model.stop_training = True





my_callback = MyThresholdCallback(threshold=0.9)



model = CnnModel()

model.compile(optimizer=Nadam(learning_rate=0.0001),loss = 'categorical_crossentropy',metrics=['accuracy'])

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,zoom_range=0.2,rotation_range=10)

datagen.fit(X_train)

iterator = datagen.flow(X_train, y_train, batch_size=32,shuffle=True)

steps = int(X_train.shape[0] / 32)

history = model.fit_generator(iterator,

                    epochs=350,verbose=1,steps_per_epoch=steps,validation_data=(X_validation,y_validation),shuffle=True,callbacks=[my_callback])'''
from keras.models import model_from_json

json_file = open("../input/finalnet/final_net.json", 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("../input/finalnet/final_net.h5")

print("Loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

score = loaded_model.evaluate(X_validation, y_validation, verbose=1)

print("Score on validation set: ", score[1])
from sklearn.metrics import confusion_matrix, classification_report

pred = loaded_model.predict(X_validation)

Y_pred_classes = np.argmax(pred, axis=1) 

Y_true = np.argmax(y_validation, axis=1)



# Errors are difference between predicted labels and true labels

errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = pred[errors]

Y_true_errors = Y_true[errors]

X_test_errors = X_validation[errors]



cm = confusion_matrix(Y_true,Y_pred_classes) 

thresh = cm.max() / 2.



cm

labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

plt.title('Confusion Matrix')

plt.figure(figsize=(10,10))

cm_df = pd.DataFrame(cm,

                     index = labels, 

                     columns = labels)

sns.heatmap(cm_df, annot=True, annot_kws={"size": 12},fmt='g')

plt.title('CNN - 89%acc',fontsize=18)

plt.ylabel('True label',fontsize=18)

plt.xlabel('Predicted label',fontsize=18)

plt.show()

#g = sns.heatmap(cm, annot=True,annot_kws={"size": 12},fmt='g')
fig, ax = plt.subplots(figsize=(10,10))

ax.matshow(cm_df, cmap=plt.cm.gray)

R = 8

C = 8

fig, axes = plt.subplots(R, C, figsize=(20,15))

axes = axes.ravel()





misclassified_idx = np.where(Y_pred_classes != Y_true)[0]

for i in np.arange(0, R*C):

    axes[i].imshow(X_validation[misclassified_idx[i]])

    axes[i].set_title("True: %s \nPredicted: %s" % (labels[Y_true[misclassified_idx[i]]], 

                                                  labels[Y_pred_classes[misclassified_idx[i]]]))

    axes[i].axis('off')

    plt.subplots_adjust(wspace=1)



print(classification_report(Y_true, Y_pred_classes))
final_testing = model.to_json()

with open("final_testing.json", "w") as json_file:

    json_file.write(model_json)

model.save_weights("final_testing.h5")

from IPython.display import FileLink, FileLinks

FileLinks('.') #lists all downloadable files on server
from IPython.display import FileLink, FileLinks

FileLinks('.') 