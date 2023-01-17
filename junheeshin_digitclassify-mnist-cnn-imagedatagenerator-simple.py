# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential, load_model

from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

from keras import backend as K



from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



from keras.preprocessing.image import ImageDataGenerator

# graph

import seaborn as sns

import matplotlib.pyplot as plt

import random

modelpath='./digit.h5'

SEED=1234

def seed_All():

    np.random.seed(SEED)

    random.seed(SEED)

    os.environ['PYTHONHASHSEED'] = str(SEED)

    config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)

    tf.set_random_seed(SEED)

    sess = tf.Session(graph=tf.get_default_graph(), config=config)

    K.set_session(sess)

seed_All()
dftrain = pd.read_csv("../input/digit-recognizer/train.csv")

dftest = pd.read_csv("../input/digit-recognizer/test.csv")
dftrain.head()

# pixel 784

# label.

y_org = dftrain['label'].values

x_org = dftrain.iloc[:,1:].values

x_test = dftest.values
print(y_org.shape, x_org.shape, x_test.shape)
sns.countplot(y_org)
# pd.Series(x_org.flatten()).value_counts()

# pd.Series(x_org[0]).value_counts()
x_org = dftrain.iloc[:,1:].values / 255.0

x_test = dftest.values / 255.0

# reshape

x_org = x_org.reshape(-1, 28, 28, 1)

x_test = x_test.reshape(-1, 28, 28, 1)

# one-hot encoding

y_org = to_categorical(dftrain['label'].values, num_classes=10)
# split

x_train, x_val, y_train, y_val = train_test_split(x_org, y_org, test_size=0.1)

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
plt.figure()

for i in range(4):

    ix = random.randint(0, len(x_train))

    plt.subplot(1,4,i+1)

    plt.title( str(np.argmax(y_train[ix])))

    plt.xticks([])

    plt.yticks([])

    plt.imshow(x_train[ix].reshape(28,28), cmap='gray')

plt.show()

model = Sequential()

model.add( Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(28,28,1)))

model.add( Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))

model.add( MaxPool2D(pool_size=(2,2)))

model.add( Dropout(0.25))



model.add( Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))

model.add( Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))

model.add( MaxPool2D(pool_size=(2,2)))

model.add( Dropout(0.25))



model.add( Flatten())

model.add( Dense(256, activation='relu'))

model.add( Dropout(0.25))

model.add( Dense(10, activation='softmax'))





model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])



lrr = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=5, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

ckpt = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True)

es = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

callbacklist = [lrr, ckpt, es]
# ref : https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

datagen = ImageDataGenerator(

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

datagen.fit(x_train)



if os.path.exists(modelpath):

    print('load model')

    model = load_model(modelpath)
epochs=100

batch_size=50



history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_val,y_val),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size

                              , callbacks=callbacklist)
if os.path.exists(modelpath):

    print('load model')

    model = load_model(modelpath) # use best model.
results = model.predict(x_test)

results = np.argmax(results, axis=1)

results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
