import os

import warnings

import numpy as np

import pandas as pd

from PIL import Image

import tensorflow as tf

import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.models import load_model

from keras.preprocessing import image

from keras import layers, optimizers 

from keras.models import Sequential, Model

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)
def plot_history(history):

    history_dict = history.history

    loss_values = history_dict['loss']

    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))



    ax1.plot(epochs, loss_values, 'bo',

             label='Training loss')

    ax1.plot(epochs, val_loss_values, 'r',

             label='Validation loss')

    ax1.set_xlabel('Epochs')

    ax1.set_ylabel('Loss')

    ax1.set_xscale('log')



    acc_values = history_dict['accuracy']

    val_acc_values = history_dict['val_accuracy']



    ax2.plot(epochs, acc_values, 'bo',

             label='Training acc')

    ax2.plot(epochs, val_acc_values, 'r',

             label='Validation acc')

    ax2.set_xlabel('Epochs')

    ax2.set_ylabel('Accuracy')

    ax2.set_xscale('log')



    plt.legend()

    plt.show()
test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

dig_mnist = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")

sample_submission = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")
print("test.shape", test.shape)

print("train.shape", train.shape)

print("dig_mnist.shape", dig_mnist.shape)

print("sample_submission.shape", sample_submission.shape)
train.head(11)
X_train = train.loc[:, train.columns!='label'].values.astype('uint8')

print("X_train.shape", X_train.shape)

y_train = train['label'].values

X_train = X_train.reshape((X_train.shape[0],28,28))

print("X_train.shape", X_train.shape)

print("y_train.shape",X_train.shape)
test.head(11)
X_test = test.loc[:,  test.columns!='id'].values.astype('uint8')

print("X_test.shape", X_test.shape)

y_id = test['id'].values

X_test = X_test.reshape((X_test.shape[0],28,28))

print("X_test.shape", X_test.shape)

print("y_id.shape", y_id.shape)
n = np.random.randint(X_train.shape[0])

plt.imshow(Image.fromarray(X_train[n]))

plt.show()

print(f'This is a {y_train[n]}')
X_train = X_train[:,:,:,None]

X_test = X_test[:,:,:,None]
print("X_train.shape", X_train.shape)

print("X_test.shape", X_test.shape)
batch_size = 32

num_epochs = 50
num_samples = X_train.shape[0]

num_classes = np.unique(y_train).shape[0]

img_rows, img_cols = X_train[0,:,:,0].shape

classes = np.unique(y_train)
print("num_samples",num_samples)

print("num_classes",num_classes)

print("img_rows",img_rows)

print("img_cols",img_cols)

print("classes",classes)
y_train = np_utils.to_categorical(y_train, num_classes)

y_train.shape
X_train_norm = X_train.astype('float32')

X_test_norm = X_test.astype('float32')

X_train_norm /= 255

X_test_norm /= 255
learning_rate_reduction=ReduceLROnPlateau(monitor='val_loss',

                                          patience=5, 

                                          verbose=1,

                                          factor=0.2

                                         )
early_stopping = EarlyStopping(monitor='val_loss', 

                               mode='min', 

                               verbose=1, 

                               patience=10

                              )
def build_model():

    model = Sequential()

    x_in = layers.Input(shape=(28, 28, 1))

    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x_in)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(512, kernel_size=(3, 3), activation='relu')(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512)(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(256)(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(128)(x)

    x = layers.BatchNormalization()(x)

    classes = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=x_in, outputs=classes)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model = build_model()
history = model.fit(X_train_norm, 

                     y_train, 

                     batch_size = batch_size,  

                     epochs = num_epochs, 

                     validation_split = 0.1,

                     shuffle = True,

                     callbacks = [learning_rate_reduction, early_stopping]

                    )
plot_history(history)
model.save('model.h5')
# model = load_model('/kaggle/input/kannada-mnist-simpe-cnn-in-keras-weight/model.h5')
pred = model.predict(X_test_norm)
pred=np.argmax(pred, axis=1)
sample_submission['label'] = pred
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)