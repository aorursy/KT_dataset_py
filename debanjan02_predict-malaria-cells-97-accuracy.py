import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input/cell_images/cell_images"))

from glob import glob

from PIL import Image

%matplotlib inline

import matplotlib.pyplot as plt

import cv2

import fnmatch

import keras

from time import sleep

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization,MaxPooling2D,Activation

from keras.optimizers import RMSprop,Adam

from tensorflow.keras.callbacks import EarlyStopping

from keras import backend as k
imagePatches_0 = glob('../input/cell_images/cell_images/Uninfected/*.png', recursive=True)

imagePatches_1 = glob('../input/cell_images/cell_images/Parasitized/*.png', recursive=True)

print(len(imagePatches_0))

print(len(imagePatches_1))
x=[]

y=[]

for img in imagePatches_0:

    full_size_image = cv2.imread(img)

    im = cv2.resize(full_size_image, (224, 224), interpolation=cv2.INTER_CUBIC)

    x.append(im)

    y.append(0)

for img in imagePatches_1:

    full_size_image = cv2.imread(img)

    im = cv2.resize(full_size_image, (224, 224), interpolation=cv2.INTER_CUBIC)

    x.append(im)

    y.append(1)

x = np.array(x)

y = np.array(y)
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 101)

y_train = to_categorical(y_train, num_classes = 2)

y_valid = to_categorical(y_valid, num_classes = 2)

del x, y
import keras

from keras.models import Sequential,Input,Model

from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, LSTM, TimeDistributed

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU 

model = Sequential()

model.add(Conv2D(32,(7,7),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.15))

model.add(Conv2D(64,(5,5),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.15))

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.15))

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.15))

model.add(GlobalAveragePooling2D())

model.add(Dense(1000, activation='relu'))

model.add(Dense(2,activation='softmax'))

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

#model.summary()
from keras.callbacks import ModelCheckpoint

mcp = ModelCheckpoint(filepath='model_check_path.hdf5',monitor="val_acc", save_best_only=True, save_weights_only=False)

hist = model.fit(x_train,y_train,batch_size = 32, epochs = 50, verbose=1,  validation_split=0.2, callbacks=[mcp])
fig = plt.figure()

ax = fig.add_subplot(111)

ax.set_facecolor('w')

ax.grid(b=False)

ax.plot(hist.history['acc'], color='red')

ax.plot(hist.history['val_acc'], color ='green')

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='lower right')

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111)

ax.set_facecolor('w')

ax.grid(b=False)

ax.plot(hist.history['loss'], color='red')

ax.plot(hist.history['val_loss'], color ='green')

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()
model.load_weights('model_check_path.hdf5')
from sklearn.metrics import classification_report

pred = model.predict(x_valid)

print(classification_report(np.argmax(y_valid, axis = 1),np.argmax(pred, axis = 1)))
def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):

    import matplotlib.pyplot as plt

    import numpy as np

    import itertools



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.grid(b=False)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(np.argmax(y_valid, axis = 1),np.argmax(pred, axis = 1))

plot_confusion_matrix(cm = cm,

                      normalize    = False,

                      cmap ='Reds',

                      target_names = ['Uninfected', 'Parasitized'],

                      title        = "Confusion Matrix")