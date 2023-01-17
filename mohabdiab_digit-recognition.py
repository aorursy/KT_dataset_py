

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



#ML libraries

from sklearn.ensemble import RandomForestClassifier





# Deep Learning libraries

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, LearningRateScheduler



%matplotlib inline

plt.style.use('ggplot')

sns.set_style('whitegrid')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape
display(train.info())



display(test.info())



display(train.head(n = 2))

display(test.head(n = 2))
features_train = train.iloc[:, 1:]

labels_train = train.iloc[:, 0:1].values
features_train = features_train / 255.0

test = test / 255.0



sampleImageIndex = 1



sampleImagePixelMap = features_train.iloc[sampleImageIndex, :].values.reshape(28, 28)

print(sampleImagePixelMap.shape)



print("The below Image should be a ", labels_train[sampleImageIndex])

g = plt.imshow(sampleImagePixelMap)
features_train = features_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)



print(features_train.shape)

print(test.shape)
print("The below Image should still be a ", labels_train[sampleImageIndex])

g = plt.imshow(features_train[sampleImageIndex, :, :, 0])
labels_train = to_categorical(labels_train, num_classes = 10)

print(labels_train[0])
X_train, X_valid, y_train, y_valid = train_test_split(features_train, labels_train, test_size = 0.1)
digitNet = Sequential()



digitNet.add(BatchNormalization(input_shape = (28, 28, 1)))

digitNet.add(Conv2D(filters = 16, kernel_size = 3, kernel_initializer = 'he_normal', activation = 'relu', padding = 'same'))

digitNet.add(MaxPool2D(pool_size = 2))

digitNet.add(BatchNormalization())



digitNet.add(Conv2D(filters = 32, kernel_size = 3, kernel_initializer= 'he_normal', activation = 'relu', padding = 'same'))

digitNet.add(MaxPool2D(pool_size = 2))

digitNet.add(BatchNormalization())



digitNet.add(Conv2D(filters = 64, kernel_size = 3, kernel_initializer = 'he_normal', activation = 'relu', padding = 'same'))

digitNet.add(MaxPool2D(pool_size = 2))

digitNet.add(BatchNormalization())



digitNet.add(Conv2D(filters = 128, kernel_size = 3, kernel_initializer = 'he_normal', activation = 'relu', padding = 'same'))

digitNet.add(MaxPool2D(pool_size = 2))

digitNet.add(BatchNormalization())



digitNet.add(GlobalAveragePooling2D())



digitNet.add(Dense(10, activation = 'softmax'))

digitNet.summary()
digitNet.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
datagen = ImageDataGenerator(

        width_shift_range=0.1,

        height_shift_range=0.1,

        zoom_range = 0.1,

        rotation_range=10,

        horizontal_flip = False,

        vertical_flip = False,

)



datagen.fit(X_train)



checkpointer = ModelCheckpoint(filepath = 'bestModel.hdf5', 

                               verbose=1, save_best_only = True)



reduce = LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)



digitNet.fit_generator(datagen.flow(X_train, y_train, batch_size = 32), 

                       steps_per_epoch = X_train.shape[0] // 32, 

          validation_data = (X_valid, y_valid), epochs = 64,

          callbacks=[checkpointer, reduce], verbose=1)
digitNet.load_weights("bestModel.hdf5")

results = digitNet.predict(test)

# select the indix with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)