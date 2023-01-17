# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
    # This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop

from sklearn.model_selection import train_test_split

from keras import  backend as K

from keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')

print(train.shape)

train.head()
test = pd.read_csv('../input/test.csv')

print(test.shape)

test.head()
X_train = (train.iloc[:,1:].values).astype('float32')

y_train = (train.iloc[:,0].values).astype('int32')

X_test = test.values.astype('float32')
train.values
test.values
y_train
X_train = X_train.reshape(X_train.shape[0],28,28)



for i in range(6,9):

    plt.subplot(330 + (i+1)) #plots the figure

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i])
#expand 1 more dimension as 1 for color channnel gray

X_train=X_train.reshape(X_train.shape[0],28,28,1)

X_train.shape
X_test = X_test.reshape(X_test.shape[0],28,28,1)

X_test.shape
#Preprocess the digits images

#Feature standarisation

#to center the data areound mean zero and unit variance
mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)



def standardize(x):

    return (x-mean_px)/std_px
#mean_px

y_train
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train)

num_classes = y_train.shape[1]

num_classes
plt.title(y_train[9])

plt.plot(y_train[9])

plt.xticks(range(10))
#seed for reproducibility

seed = 43

np.random.seed(seed)
from keras.models import Sequential

from keras.layers.core import Lambda, Dense, Flatten, Dropout

from keras.callbacks import EarlyStopping

from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D
#Linear Model

#layer different
model = Sequential()

model.add(Lambda(standardize, input_shape = (28,28,1))) #this layer will take given dimension as input

model.add(Flatten())#mak it t 1D

model.add(Dense(10,activation='softmax')) # #type of output and layer will sure that all neurons are connected to previous

print("input shape",model.input_shape)

print("output shape",model.output_shape)
from keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),

             loss='categorical_crossentropy',

             metrics=['accuracy'])
from keras.preprocessing import image

gen = image.ImageDataGenerator()
# from sklearn.model_selection import train_test_split

X =X_train

y = y_train

X_train, X_val, y_train,y_val = train_test_split(X_train, y_train,

                                                test_size=0.10, random_state=42)

batches = gen.flow(X_train,y_train, batch_size=64)

val_batches = gen.flow(X_val, y_val, batch_size=64)
history = model.fit_generator(generator=batches, steps_per_epoch = batches.n,

                             epochs=3, validation_data=val_batches,

                             validation_steps=val_batches.n)
history_dict=history.history

history_dict.keys()
history
import matplotlib.pyplot as plt

%matplotlib inline

loss_values = history_dict["loss"]

val_loss_values=history_dict["val_loss"]

epochs=range(1,len(loss_values)+1)

plt.plot(epochs, loss_values, 'bo')

plt.plot(epochs, val_loss_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Loss')



plt.show()
plt.clf()

acc_values=history_dict['acc']

val_acc_values=history_dict['val_acc']



plt.plot(epochs, acc_values,'bo')

plt.plot(epochs, val_acc_values,'b+')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.show()
#fully connected model
# def get_fc_model():

#     model = Sequential([

#         Lambda(standardize, input_shape=(28,28,1)),

#         Flatten(),

#         Dense(512, activation='relu'),

#         Dense(10, activation='softmax')

#     ])

#     model.compile(optimizer='Adam', loss='categorical_crossentropy',

#                  metrics=['accuracy'])

#     return model

# fc = get_fc_model()

# fc.optimizer.lr=0.01
# history=fc.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

#                     validation_data=val_batches, validation_steps=val_batches.n)
#Convolutional Neural Network
from keras.layers import Convolution2D,MaxPooling2D

def get_cnn_model():

    model=Sequential([

        Lambda(standardize, input_shape=(28,28,1)),

        Convolution2D(32,(3,3), activation='relu'),

        Convolution2D(32,(3,3), activation='relu'),

        MaxPooling2D(),

        Convolution2D(64,(3,3), activation='relu'),

        Convolution2D(64,(3,3), activation='relu'),

        MaxPooling2D(),

        Flatten(),

        Dense(512, activation='relu'),

        Dense(10, activation='softmax')

    ])

    model.compile(Adam(), loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model
model = get_cnn_model()

model.optimizer.lr=0.01
history = model.fit_generator(generator=batches, steps_per_epoch=batches.n,

                             epochs=1,validation_data=val_batches,

                             validation_steps=val_batches.n)
#Data Augmentation




# gen =ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,

#                                height_shift_range=0.08, zoom_range=0.08)

# batches = gen.flow(X_train, y_train, batch_size=64)

# val_batches = gen.flow(X_val, y_val, batch_size=64)







# model.optimizer.lr=0.001

# history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

#                     validation_data=val_batches, validation_steps=val_batches.n)
#Adding Batch Normalisation
# from keras.layers.normalization import BatchNormalization



# def get_bn_model():

#     model = Sequential([

#         Lambda(standardize, input_shape=(28,28,1)),

#         Convolution2D(32,(3,3), activation='relu'),

#         BatchNormalization(axis=1),

#         Convolution2D(32,(3,3), activation='relu'),

#         MaxPooling2D(),

#         BatchNormalization(axis=1),

#         Convolution2D(64,(3,3), activation='relu'),

#         BatchNormalization(axis=1),

#         Convolution2D(64,(3,3), activation='relu'),

#         MaxPooling2D(),

#         Flatten(),

#         BatchNormalization(),

#         Dense(512, activation='relu'),

#         BatchNormalization(),

#         Dense(10, activation='softmax')

#         ])

#     model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

#     return model
# model= get_bn_model()

# model.optimizer.lr=0.01

# history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

#                     validation_data=val_batches, validation_steps=val_batches.n)
# model.optimizer.lr=0.01

# gen = image.ImageDataGenerator()

# batches = gen.flow(X, y ,batch_size=64)

# history = model.fit_generator(generator=batches, steps_per_epoch=batches.n,

#                              epochs=3)
predictions=model.predict_classes(X_test, verbose=0)

submissions = pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),

                           "Label":predictions})

submissions.to_csv("c_m.csv", index=False, header=True)