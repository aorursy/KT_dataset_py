import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.model_selection import train_test_split

import itertools

import math

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, ZeroPadding2D, Input

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.applications.resnet50 import ResNet50

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import VGG16



%matplotlib inline



pd.set_option("display.max_rows", 6)



np.random.seed(2)
!ls ../input
# Load the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train_X = (train.iloc[:,:-1]/255).values.copy()

train_y = (train['label']).values.copy()

test = (test/255).values.copy()



print(train_X.shape)

print(train_y.shape)

print(test.shape)

train_X = np.reshape(train_X, (42000, 28,28,1))

new_X = np.zeros((42000,28,28,3))

for i in range(len(new_X)):

    tmp = np.stack((train_X[i],)*3, axis=-1)

    new_X[i] = np.resize(tmp, (28,28,3))

train_X = new_X





test = np.reshape(test, (28000, 28,28,1))

new_X = np.zeros((28000,28,28,3))

for i in range(len(new_X)):

    tmp = np.stack((test[i],)*3, axis=-1)

    new_X[i] = np.resize(tmp, (28,28,3))

test = new_X





from keras.utils import to_categorical

train_y = to_categorical(train_y, num_classes=10)

batch_size = 4

datagen = ImageDataGenerator(

    validation_split=.2,

    rotation_range=10,

    width_shift_range=0.2,

    height_shift_range=0.2,

    fill_mode='nearest')



train_generator = datagen.flow(

        train_X,

        train_y,

        shuffle=True,

        subset='training',

        batch_size=batch_size,)

val_generator = datagen.flow(

        train_X,

        train_y,

        subset='validation',

        batch_size=batch_size,)





plt.figure(figsize=(15,10)) 

for X_batch, y_batch in train_generator:

    for i in range(0, 4):

        plt.subplot(220 + 1 + i)

        plt.imshow(X_batch[i].reshape(28,28,3), cmap=plt.get_cmap('gray'))

    plt.show()

    break

#g = plt.imshow(train_X[0][:,:,0])


conv_base = VGG16(#weights='imagenet',

                  include_top=False,

                  input_shape=(36, 36, 3))



#for layer in conv_base.layers[:-4]:

#   layer.trainable = False

#for layer in conv_base.layers:

#    print(layer, layer.trainable)



for layer in conv_base.layers:

   layer.trainable = True



model = Sequential()

model.add(ZeroPadding2D(padding=(32-28, 32-28), input_shape=(28,28,3)))

model.add(conv_base)

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(10, activation='softmax'))

from keras.optimizers import SGD, Adam



# I manually change the lr to 1e-3 when it gets stuck on .98

model.compile(optimizer=SGD(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

from keras.models import load_model



model = load_model('vgg16_dense.h5')
epochs = 1 # change epochs to improve



model.fit_generator(train_generator, 

                    validation_data=val_generator, 

                    validation_steps=(42000*.2)//batch_size, 

                    epochs=epochs, 

                    steps_per_epoch=(42000-42000*.2)//batch_size)

model.save('vgg16_dense.h5')
pred = model.predict(test)

pred.shape
plt.figure(figsize=(4,4)) 

plt.imshow(test[100].reshape(28,28,3), cmap=plt.get_cmap('gray'))

out_label = [ np.argmax(i) for i in pred]

out_imageid = [ i+1 for i in range(len(test))]

out = pd.DataFrame()

out['ImageId'] = out_imageid

out['Label'] = out_label



out.to_csv('submission.csv', index=False)

out