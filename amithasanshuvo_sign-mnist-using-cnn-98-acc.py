import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd
train=pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')

test=pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')
train.head()
train.info()
test.info()
train.shape
test.shape
labels = train['label'].values

plt.figure(figsize = (14,8))

sns.countplot(x =labels)


X_train = train.drop(["label"],axis=1)

X_test = test.drop(["label"],axis=1)

Y_train = train['label']

Y_test = test['label']

del train['label']

del test['label']
X_train = X_train/255.0

X_test = X_test/255.0

X_train.shape

X_test.shape
X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

X_train.shape

X_test.shape
from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()

Y_train = label_binrizer.fit_transform(Y_train)
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)



print("x_train shape",X_train.shape)

print("x_test shape",X_val.shape)

print("y_train shape",Y_train.shape)

print("y_test shape",Y_val.shape)
from sklearn.metrics import confusion_matrix

import itertools

import tensorflow as tf



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau





model = tf.keras.models.Sequential([

                        tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding = 'same', input_shape=(28, 28, 1)),

                        tf.keras.layers.MaxPooling2D(2, 2),

                        tf.keras.layers.Conv2D(128, (3, 3),padding = 'same', activation='relu'),

                        tf.keras.layers.MaxPooling2D(2, 2),

                        tf.keras.layers.Conv2D(512, (3, 3),padding = 'same', activation='relu'),

                        tf.keras.layers.MaxPooling2D(2, 2),

                        

                        tf.keras.layers.Conv2D(512, (3, 3),padding = 'same', activation='relu'),

                        tf.keras.layers.Flatten(),

                        tf.keras.layers.Dense(512, activation='relu'),

                        tf.keras.layers.Dense(24, activation='softmax')])

model.summary()
#optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999)



model.compile( optimizer='rmsprop' , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 50 

batch_size = 200

#Data augmentation

datagen = ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False, 

        zca_whitening=False,

        rotation_range=15, 

        zoom_range = 0.5,

        width_shift_range=0.15,  

        height_shift_range=0.15, 

        horizontal_flip=True,  

        vertical_flip=False)  



datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()