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
Y = train['label'].values



X_exp = []

for i in tnrange(train.shape[0]):

    im = train.iloc[i][train.columns[1:]].values.reshape((28,28))

    newim = np.zeros((32,32))

    newim[2:-2,2:-2] = im

    X_exp.append(newim)

X = np.array(X_exp)/255.

X = X.reshape(X.shape[0],32,32,1)



X_exp = []

for i in tnrange(test.shape[0]):

    im = test.iloc[i][test.columns[1:]].values.reshape((28,28))

    newim = np.zeros((32,32))

    newim[2:-2,2:-2] = im

    X_exp.append(newim)

X_valid = np.array(X_exp)/255.

X_valid = X_valid.reshape(X_valid.shape[0],32,32,1)
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
# CNN architechture

f = 2**2



model = tf.keras.Sequential([

    # layer 1

    tf.keras.layers.Conv2D(f*16,kernel_size=(3,3),padding="same",activation='relu',

                           kernel_initializer='he_uniform', 

                           input_shape=(32,32,1)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(f*16, (3,3), padding='same', 

                           activation ='relu',

                           kernel_regularizer=regularizers.l2(0.01)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(f*16, (5,5), padding='same', activation ='relu',

                          kernel_regularizer=regularizers.l2(0.01)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),

    #tf.keras.layers.Dropout(0.15),

    

    tf.keras.layers.Conv2D(f*32, (3,3), padding='same', activation ='relu',

                          kernel_regularizer=regularizers.l2(0.01)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(f*32, (3,3), padding='same', activation ='relu',

                          kernel_regularizer=regularizers.l2(0.01)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(f*32, (5,5), padding='same', activation ='relu',

                          kernel_regularizer=regularizers.l2(0.01)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.15),

    

    # layer 3

    tf.keras.layers.Conv2D(f*64,kernel_size=(3,3),padding="same",activation='relu',

                          kernel_regularizer=regularizers.l2(0.01)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(f*64,kernel_size=(3,3),padding="same",activation='relu',

                          kernel_regularizer=regularizers.l2(0.01)),

    #tf.keras.layers.Conv2D(f*64,kernel_size=(5,5),padding="same",activation='relu'),

    #tf.keras.layers.Conv2D(f*64,kernel_size=(5,5),padding="same",activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.15),

    

    # layer 4

    #tf.keras.layers.Conv2D(f*128,kernel_size=(3,3),padding="same",activation='relu'),

    #tf.keras.layers.Conv2D(f*128,kernel_size=(3,3),padding="same",activation='relu'),

    #tf.keras.layers.BatchNormalization(),

    #tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'), #512

    tf.keras.layers.Dense(128, activation='relu'),

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
from scipy.fftpack import dct, idct



def dct2(block):

    return dct(dct(block.T, norm='ortho').T, norm='ortho')



def idct2(block):

    return idct(idct(block.T, norm='ortho').T, norm='ortho')



class MyIterator(Iterator):

  """This is a toy example of a wrapper around ImageDataGenerator"""



  def __init__(self, x, y, batch_size, shuffle, seed, **kwargs):

    super().__init__(x.shape[0], batch_size, shuffle, seed)



    # Load any data you need here (CSV, HDF5, raw stuffs). The code

    # below is just a pseudo-code for demonstration purpose.

    self.input_images = x

    self.ground_truth = y



    # Here is our beloved image augmentator <3

    self.generator = ImageDataGenerator(**kwargs)



  def _get_batches_of_transformed_samples(self, index_array):

    """Gets a batch of transformed samples from array of indices"""



    # Get a batch of image data

    batch_x = self.input_images[index_array].copy()

    batch_y = self.ground_truth[index_array].copy()



    # Transform the inputs and correct the outputs accordingly

    for i, (x, y) in enumerate(zip(batch_x, batch_y)):

        transform_params = self.generator.get_random_transform(x.shape)

        batch_x[i] = self.generator.apply_transform(x, transform_params)

        batch_x[i] = dct2(batch_x[i].reshape((32,32))).reshape((32,32,1))/1500 

        batch_y[i] = y

        

    return batch_x, batch_y
batchsize = 200

epoch = 45

callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)

train_datagen = MyIterator(X,

                           Y,

                           horizontal_flip=False,

                           rotation_range=15,

                           width_shift_range=0.25,

                           height_shift_range=0.25,

                           shear_range=0.15,

                           zoom_range=0.25,batch_size = batchsize,shuffle=True,seed=0)

val_datagen = MyIterator(X_dev,Y_dev,batch_size = batchsize,shuffle=False,seed=0)

history = model.fit(train_datagen,

                   steps_per_epoch = 100, 

                    epochs = epoch,

                   callbacks=[callback,

                            LearningRateScheduler(lr_decay),

                            lr

                             ],

                   validation_data=val_datagen,

                   validation_steps=50,

                   )
val_datagen = MyIterator(X_valid,submission.label.values,batch_size = batchsize,shuffle=False,seed=0)



yhat = model.predict_generator(val_datagen).argmax(axis=1)

submission['label']=pd.Series(yhat)

submission.to_csv('submission.csv',index=False)
submission.head()
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import h5py



from keras.models import load_model



model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'