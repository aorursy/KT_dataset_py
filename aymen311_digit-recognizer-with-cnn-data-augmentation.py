import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import matplotlib.image as mpimg

import seaborn as sns 

%matplotlib inline 



np.random.seed(0)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools 



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



sns.set(style = 'white', context = 'notebook', palette = 'deep')
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
Y_train = train["label"]



X_train = train.drop(labels = ["label"], axis = 1)

del train 



Y_train.value_counts()

X_train 
g = sns.countplot(Y_train)
X_train.isnull().any().describe() 
test.isnull().any().describe()
X_train = X_train / 255.0

test = test / 255.0

X_train.shape
X_train = np.reshape(X_train.values, (-1, 28, 28, 1))

X_train.shape

test = np.reshape(test.values, (-1, 28, 28, 1))

test.shape
Y_train = to_categorical(Y_train, num_classes = 10)

Y_train[1]
random_seed = 2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = random_seed)

X_train.shape
idx = 5

g = plt.imshow(X_train[idx][:,:,0])
model = Sequential()



model.add(Conv2D(filters = 32 , kernel_size = (5,5), activation = 'relu', padding = 'Same', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu', padding = 'Same'))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64 , kernel_size = (3,3), activation = 'relu', padding = 'Same'))

model.add(Conv2D(filters = 32 , kernel_size = (3,3), activation = 'relu', padding = 'Same'))

model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation = 'softmax' ))

optimizer = RMSprop(lr = 0.001, decay = 0.0, epsilon = 1e-08)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose = 1, min_lr = 0.0001)
epochs = 30

batch_size = 86
#history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose = 2, validation_data = (X_val ,Y_val))
datagen = ImageDataGenerator (featurewise_center = False,

                              samplewise_center = False, 

                              featurewise_std_normalization = False,

                              samplewise_std_normalization = False,

                              zca_whitening=False,

                              rotation_range = 10,

                              zoom_range = 0.1,

                              width_shift_range = 0.1,

                              height_shift_range = 0.1,

                              horizontal_flip = False,

                              vertical_flip=False)

datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size),epochs = epochs,

 validation_data = (X_val, Y_val),verbose = 1,steps_per_epoch = X_train.shape[0] // batch_size,

                                           callbacks = [learning_rate_reduction])
fig,ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], label = "Training loss", color ="b")

ax[0].plot(history.history['val_loss'], label = " Validation loss", color ="yellow", axes = ax[0])

legend = ax[0].legend(loc = 'best', shadow =True)



ax[1].plot(history.history['accuracy'], label = "Training accuracy", color = 'b')

ax[1].plot(history.history['val_accuracy'], label = "Validation accuracy", color = 'yellow')

legend = ax[1].legend(loc = 'best', shadow =True)
# predict results

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)