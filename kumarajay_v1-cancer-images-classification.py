# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

from PIL import Image

%matplotlib inline

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/paths-csv"))



# Any results you write to the current directory are saved as output.
df= pd.read_csv("../input/paths-csv/file_paths.csv")
df.head()
df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((224,224))))
df.head()
from sklearn.model_selection import train_test_split
features=df['image']

target=df['target']
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20,random_state=1234)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.20,random_state=1234)
#x_valid=x_test

#y_valid=y_test
print(len(x_train))

print(len(x_valid))

print(len(x_test))

y_train[0:4]
x_train = np.asarray(x_train.tolist())

x_valid = np.asarray(x_valid.tolist())

x_test = np.asarray(x_test.tolist())



x_train_mean = np.mean(x_train)

x_train_std = np.std(x_train)



x_valid_mean = np.mean(x_valid)

x_valid_std = np.std(x_valid)



x_test_mean = np.mean(x_test)

x_test_std = np.std(x_test)



x_train = (x_train - x_train_mean)/x_train_std

x_valid = (x_valid - x_valid_mean)/x_valid_std

x_test = (x_test - x_test_mean)/x_test_std
x_train = x_train.reshape(x_train.shape[0], *(224, 224, 3))

x_test = x_test.reshape(x_test.shape[0], *(224, 224, 3))

x_valid = x_valid.reshape(x_valid.shape[0], *(224, 224, 3))
from keras.utils.np_utils import to_categorical 

y_train = to_categorical(y_train, num_classes = 4)

y_valid = to_categorical(y_valid, num_classes = 4)

y_test = to_categorical(y_test, num_classes = 4)
y_valid.shape
from PIL import Image

from sklearn.preprocessing import label_binarize

from sklearn.metrics import confusion_matrix

import itertools



import keras

from keras.utils.np_utils import to_categorical 

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras import backend as K

import itertools

from keras.layers.normalization import BatchNormalization

from keras.utils.np_utils import to_categorical 



from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping

from sklearn.model_selection import train_test_split







import cv2

import numpy as np

import pandas as pd

from sklearn.model_selection import KFold

from sklearn import metrics

import keras

from keras.models import Model

from keras.optimizers import Adam

from keras.applications.inception_v3 import InceptionV3

from keras.layers import Dense, Input, Flatten, Dropout, GlobalAveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau





img_height = 224

img_width = 224

img_channels = 3

img_dim = (img_height, img_width, img_channels)



def inceptionv3(img_dim=img_dim):

    input_tensor = Input(shape=img_dim)

    base_model = InceptionV3(include_top=False,

                   weights='imagenet',

                   input_shape=img_dim)

    bn = BatchNormalization()(input_tensor)

    x = base_model(bn)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)

    output = Dense(4, activation='sigmoid')(x)

    model = Model(input_tensor, output)

    return model



model = inceptionv3()

model.summary()
'''

input_shape = (224, 224, 3)

num_classes = 4



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))

model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',))

model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Dropout(0.20))



model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.20))



#model.add(Conv2D(128, (3, 3), activation='relu',padding = 'Same'))

#model.add(Conv2D(128, (3, 3), activation='relu',padding = 'Same'))

#model.add(MaxPool2D(pool_size=(2, 2)))

#model.add(Dropout(0.20))



model.add(Conv2D(128, (3, 3), activation='relu',padding = 'Same'))

model.add(Conv2D(128, (3, 3), activation='relu',padding = 'Same'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.20))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

model.summary()

'''
model.compile(optimizer = "Adam" , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=7, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

model_check_point = ModelCheckpoint(filepath="best_model.h5", verbose=1, save_best_only=True)



early_stopping=EarlyStopping(monitor='val_acc', patience=10, verbose=1)
datagen = ImageDataGenerator(

        rotation_range=25,

        zoom_range = 0.1, 

        width_shift_range=0.1,  

        height_shift_range=0.1,  

        horizontal_flip=True,  

        vertical_flip=True)  



datagen.fit(x_train)
y_train
# Fit the model

epochs = 100

batch_size = 10

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_valid,y_valid),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction,model_check_point,early_stopping])

from keras.models import load_model

model = load_model('best_model.h5')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

loss_v, accuracy_v = model.evaluate(x_valid, y_valid, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

# Function to plot confusion matrix    

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



Y_pred = model.predict(x_valid)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(y_valid,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)



plot_confusion_matrix(confusion_mtx, classes = range(4)) 
Y_pred = model.predict(x_test)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(y_test,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

plot_confusion_matrix(confusion_mtx, classes = range(4)) 
def plot_model_history(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy

    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])

    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)

    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()

plot_model_history(history)