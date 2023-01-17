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
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
df_train.head()
df_test.head()
print("train",df_train.shape)
print("test", df_test.shape)
train_y = df_train['label']
df_train.drop('label', 1, inplace = True)
print("train", df_train.shape)
X=df_train.iloc[:,:-1].values
Y=df_train['label'].values
X=X/255.0
df_test=df_test/255.0
X = X.reshape(-1,28,28,1)
df_test=df_test.values.reshape(-1,28,28,1)
Y=Y.reshape(-1,1)
Y
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, Input, Convolution2D, BatchNormalization
import pandas as pd
import os
import numpy as np
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
import cv2
import time
Y = to_categorical(Y, num_classes = 10)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.05,random_state=90)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.05, random_state=90)
class CNNmodel:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
 
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Convolution2D(32, (3, 3), input_shape=inputShape,activation='relu',padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(Convolution2D(32, (3, 3), activation='relu',padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 16x16x3xnum_filters
        model.add(Dropout(0.2))

        # second set of CONV => RELU => POOL layers
        model.add(Convolution2D(2*48, (3, 3), activation='relu',padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(Convolution2D(2*48, (3, 3), activation='relu',padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 4x4x3x(4*num_filters)
        model.add(Dropout(0.2))

        # first (and only) set of FC => RELU layers
        model.add(Convolution2D(64, (3, 3), activation='relu',padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(Convolution2D(64, (3, 3), activation='relu',padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 4x4x3x(4*num_filters)
        model.add(Dropout(0.2))
 
        # softmax classifier
        
        model.add(Convolution2D(128, (3, 3), activation='relu',padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(Convolution2D(128, (3, 3), activation='relu',padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 4x4x3x(4*num_filters)
        model.add(Dropout(0.2))

        model.add(Convolution2D(256, (3, 3), activation='relu',padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(Convolution2D(256, (3, 3), activation='relu',padding='same'))
        model.add(BatchNormalization(axis=-1))
        # reduces to 4x4x3x(4*num_filters)
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(units=256,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(10, activation = "softmax"))
        # return the constructed network architecture
        return model
EPOCHS = 50

aug = ImageDataGenerator(
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


aug.fit(x_train)
model = CNNmodel.build(width=28, height=28, depth=1, classes=10)
model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])

# train the network
H = model.fit_generator(aug.flow(x_train,y_train, batch_size=32),
                              epochs = 50, validation_data = (x_val,y_val),
                              verbose = 1)
def plot_graph(H,EPOCHS,INIT_LR,BS):

    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on our system")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    # plt.savefig(args["plot"])
    plt.show()
plot_graph(H,EPOCHS,0.001,32)
score=model.evaluate(x_test,y_test,verbose=1)               #evaluates the model
accuracy=100*score[1]                                       
print('Test accuracy is %.4f%%' % accuracy) 
score=model.evaluate(x_val,y_val,verbose=1)               #evaluates the model
accuracy=100*score[1]                                       
print('Test accuracy is %.4f%%' % accuracy)                 #returns the accuracy of the model 
pred = model.predict(df_test)
pred = pd.DataFrame(pred)
pred['Label'] = pred.idxmax(axis=1)
pred.head(5)
pred['index'] = list(range(1,len(pred)+1))
pred.head()
submission = pred[['index','Label']]
submission.head()

submission.rename(columns={'index':'ImageId'},inplace = True)
submission.head()
submission.to_csv('submission.csv',index=False)
pd.read_csv('submission.csv')
