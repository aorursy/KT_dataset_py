import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator, Iterator

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow import keras

from keras import regularizers

from tensorflow.keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import LearningRateScheduler

import pywt

from tqdm import tnrange, tqdm_notebook

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/Kannada-MNIST/train.csv")

test = pd.read_csv("../input/Kannada-MNIST/test.csv")

submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")
print("train shape is: " + str(train.shape))

print("test shape is: " + str(test.shape))
test.head()
X = train.drop(['label'], axis = 1)

X_valid = test.drop(['id'], axis = 1)
print("original TRAIN shape: " + str(X.shape))

print("original TEST shape: " + str(X_valid.shape))
def haar(block):

    a = pywt.dwt2(block, 'db1')

    return a
Y = train['label'].values



X_exp = []

for i in tnrange(train.shape[0]):

    im = train.iloc[i][train.columns[1:]].values.reshape((28,28))

    a,(b,c,d) = haar(im)

    newim = np.zeros((14,14,4))

    newim[:,:,0] = a

    newim[:,:,1] = b

    newim[:,:,2] = c

    newim[:,:,3] = d

    X_exp.append(newim)

X = np.array(X_exp)

X_exp = []

for i in tnrange(test.shape[0]):

    im = test.iloc[i][test.columns[1:]].values.reshape((28,28))

    a,(b,c,d) = haar(im)

    newim = np.zeros((14,14,4))

    newim[:,:,0] = a

    newim[:,:,1] = b

    newim[:,:,2] = c

    newim[:,:,3] = d

    X_exp.append(newim)

X_valid = np.array(X_exp)

from sklearn.model_selection import train_test_split

X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size = 0.2)
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X[i][:,:,0], cmap=plt.cm.binary)

    plt.xlabel(train.label[i])

plt.show()
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3,3), padding='same', input_shape=(14, 14, 4)),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(64,  (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(64,  (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),



    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.25),

    

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),    

    

    tf.keras.layers.Conv2D(256, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(256, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),##

    tf.keras.layers.LeakyReLU(alpha=0.1),



    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256),

    tf.keras.layers.LeakyReLU(alpha=0.1),

 

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(10, activation='softmax')

])



print(model.summary())



initial_learningrate=0.001#*0.3

model.compile(optimizer=

              #Adam(learning_rate=0.0003),

              RMSprop(lr=initial_learningrate),

             loss = 'sparse_categorical_crossentropy',

             metrics = ['accuracy'])
lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', 

                                            patience=300, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
def lr_decay(epoch, initial_learningrate = 0.001):#lrv 0.0003

    return initial_learningrate * 0.99 ** epoch
batchsize = 200

epoch = 45

train_datagen = ImageDataGenerator(#rescale=1./255.,

                                   rotation_range=10,

                                   width_shift_range=0.25,

                                   height_shift_range=0.25,

                                   shear_range=0.1,

                                   zoom_range=0.25,

                                   horizontal_flip=False)









valid_datagen = ImageDataGenerator(#rescale=1./255.,

                                    horizontal_flip=False,

                                    rotation_range=15,

                                   width_shift_range=0.25,

                                   height_shift_range=0.25,

                                   shear_range=0.15,

                                   zoom_range=0.25,

                                    )





# add early stopping

callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)



# fit model with generated data





history = model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size = batchsize),

                   steps_per_epoch = 100, 

                    epochs = epoch,

                   callbacks=[callback,

                            LearningRateScheduler(lr_decay),

                            lr

                             ],

                   validation_data=valid_datagen.flow(X_dev, Y_dev),

                    validation_steps=50,

                   )
yhat = model.predict(X_valid).argmax(axis=1)

submission['label']=pd.Series(yhat)

submission.to_csv('submission.csv',index=False)
submission.head()
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import h5py



from keras.models import load_model



model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'