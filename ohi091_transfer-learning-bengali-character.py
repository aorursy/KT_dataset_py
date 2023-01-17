import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df1 = pd.read_csv('../input/new_dataset/ekushCSV/Female/femaleCharacters.csv')

df2 = pd.read_csv('../input/new_dataset/ekushCSV/Male/malechar1.csv')

df3 = pd.read_csv('../input/new_dataset/ekushCSV/Male/malechar2.csv')
train = pd.concat([df1, df2, df3])

print(train.shape)

train.head()
labels = train['label'].values

unique_val = np.array(labels)

np.unique(unique_val)
plt.figure(figsize = (18,8))

sns.countplot(x =labels)
train.drop('label', axis = 1, inplace = True)

images = train.values

images = np.array([np.reshape(i, (28, 28)) for i in images])

images = np.array([i.flatten() for i in images])
from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()

labels = label_binrizer.fit_transform(labels)
labels
plt.imshow(images[0].reshape(28,28), cmap='gray')
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, stratify=labels, random_state = 101)
import keras

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

from keras.applications import VGG16;

from keras.applications.vgg16 import preprocess_input
x_train = x_train / 255

x_test = x_test / 255
x_train_t = np.stack([x_train.reshape(x_train.shape[0],28,28)]*3, axis=3).reshape(x_train.shape[0],28,28,3)

x_test_t = np.stack([x_test.reshape(x_test.shape[0],28,28)]*3, axis=3).reshape(x_test.shape[0],28,28,3)

x_train_t.shape, x_test_t.shape
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train.shape, x_test.shape
plt.imshow(x_train_t[0].reshape(28,28,3))
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



# Set the CNN model 

# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



model = Sequential()



#1

model.add(Conv2D(filters = 16, kernel_size = (2,2),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))



#2

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# model.add(Dropout(0.25))





#3

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# model.add(Dropout(0.5))



#4

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))







model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dense(50, activation = "softmax"))

# Define the optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

epochs=50

batch_size=86

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        #rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(x_train)
# # Fit the model

history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=20, batch_size=128)
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='r',label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='b', label="validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
plt.plot(history.history['acc'], color='r',label="Training accuracy")

plt.plot(history.history['val_acc'], color='b', label="validation accuracy")
from keras.models import load_model



model.save('basic_model.h5') 