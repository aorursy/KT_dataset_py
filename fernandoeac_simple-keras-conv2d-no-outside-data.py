!pip install neural-structured-learning

!pip install tensorflow-gpu==2.1.0
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization

from tensorflow.keras.optimizers import Adam, Nadam



import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





#Adversarial Learning

import neural_structured_learning as nsl



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
tf.executing_eagerly()
train = pd.read_csv('/kaggle/input/train.csv')

train.head()
def get_labels(train):

    return np.array(train['label'])

    #return tf.keras.utils.to_categorical(train['label'])

    

def get_features(train):

    features = train.drop(['label'], axis=1)

    features_normalized = features / 255

    

    return features_normalized
def get_model(shape, n_class):

    model = Sequential()

    model.add(Conv2D(64, (4,4), input_shape=shape))

    model.add(MaxPool2D((2,2)))

    model.add(BatchNormalization())

    

    model.add(tf.keras.layers.Activation("selu")),

    model.add(Conv2D(128, (3,3)))

    model.add(MaxPool2D((2,2)))

    model.add(BatchNormalization())

    

    model.add(tf.keras.layers.Activation("selu")),

    model.add(Conv2D(256, (2,2)))

    model.add(MaxPool2D((2,2)))

    model.add(Dropout(0.2))

    model.add(BatchNormalization())

    

    model.add(Flatten())

    model.add(tf.keras.layers.Activation("selu"))

    model.add(Dense(64, kernel_initializer="he_normal", use_bias=False))

    model.add(Dropout(0.2))

    model.add(BatchNormalization())

    

    model.add(tf.keras.layers.Activation("selu"))

    model.add(Dense(32, kernel_initializer="he_normal", use_bias=False))

    model.add(BatchNormalization())

    

    model.add(Dense(n_class, activation='softmax'))

    

#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    return model

    
def get_adv_model(shape, n_class):

    model = Sequential([

        Conv2D(64, (4,4), input_shape=shape, dynamic=True),

        MaxPool2D((2,2)),

        BatchNormalization(),

        

        tf.keras.layers.Activation("selu"),

        Conv2D(128, (3,3)),

        MaxPool2D((2,2)),

        BatchNormalization(),

    

        tf.keras.layers.Activation("selu"),

        Conv2D(256, (2,2)),

        MaxPool2D((2,2)),

        Dropout(0.2),

        BatchNormalization(),

    

        Flatten(),

        tf.keras.layers.Activation("selu"),

        Dense(64, kernel_initializer="he_normal", use_bias=False),

        Dropout(0.2),

        BatchNormalization(),



        tf.keras.layers.Activation("selu"),

        Dense(32, kernel_initializer="he_normal", use_bias=False),

        BatchNormalization(),



        Dense(n_class, activation='softmax')])

    

#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.05)

    adv_model = nsl.keras.AdversarialRegularization(model, adv_config=adv_config)

    

    adv_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)

    

    return adv_model

    
def get_sample_model(shape, n_class):

    model = tf.keras.Sequential([

        tf.keras.Input(shape, name='feature'),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation=tf.nn.relu),

        tf.keras.layers.Dense(n_class, activation=tf.nn.softmax)

    ])

    

    adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.05)

    adv_model = nsl.keras.AdversarialRegularization(model, adv_config=adv_config)



    # Compile, train, and evaluate.

    adv_model.compile(optimizer='adam',

                      loss='sparse_categorical_crossentropy',

                      metrics=['accuracy'])

    

    return adv_model
features = get_features(train)

labels = get_labels(train)

x = np.array(features).reshape(-1, 28,28,1)

x.shape
plt.imshow(x[1].reshape(28,28))
model = get_adv_model(x[0].shape, 10)

print(model)
train_x = x[:36000]

test_x = x[36000:]

labels_train = labels[:36000]

labels_test = labels[36000:]
#following https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

from tensorflow.keras.preprocessing.image import ImageDataGenerator



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



datagen.fit(train_x)
earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10)

#following https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

reduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 

                  patience=3, 

                  verbose=1, 

                  factor=0.5, 

                  min_lr=0.00001)
model.fit({'feature': train_x, 'label': np.array(labels_train)}, batch_size=32, epochs=5)

#model.fit(datagen.flow(train_x, labels_train, batch_size=32),

#          steps_per_epoch=len(train_x) // 32,

#          validation_data=(test_x, labels_test),

#          epochs=1000,

#          callbacks=[earlyStop, reduceLROnPlateau]

#)
model.evaluate({'feature': test_x, 'label': np.array(labels_test)})
test = pd.read_csv('/kaggle/input/test.csv')

test.head()
test_normalized = test/ 255



x_test = np.array(test_normalized).reshape(-1,28,28,1)
lo = [0] * len(x_test)

predictions = model.predict({'feature':x_test, 'label':np.array(lo)})
response = []

for i, prediction in enumerate(predictions):

    response.append([i + 1, np.argmax(prediction)])

    

resposta = pd.DataFrame(response, columns=['ImageId', 'Label'])

resposta.to_csv('output.csv', index=False)
