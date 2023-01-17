%matplotlib inline

%config IPCompleter.greedy=True



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



import tensorflow as tf

import tensorflow_hub as hub

from tensorflow import keras

from tensorflow.keras import datasets, layers, models

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.optimizers import Adam



from keras.utils import Sequence

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler



import warnings

warnings.filterwarnings('ignore')
# Each image is 28 pixels in height and 28 pixels in width

# total of 784 pixels
# test = pd.read_csv("../../data/digitrecognizer/test.csv")

# train = pd.read_csv("../../data/digitrecognizer/train.csv")



test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
train.info(), test.info()
# split label and feature data

train_labels = (train['label'])

train_features = (train.iloc[:, 1:].values).astype('float32')

test_features = (test.values).astype('int32')
# reshape into pixel dimensions and 1 column for gray scale

train_features = train_features.reshape(train_features.shape[0], 28, 28, 1)

test_features = test_features.reshape(test_features.shape[0], 28, 28, 1)

train_features.shape, test_features.shape, train_labels.shape
# display sample digits 

for i in range(16, 7):

    plt.subplot(330 + (i+1))

    plt.imshow(train_features[i], cmap=plt.get_cmap('gray'))

    plt.title(train_labels[i]);



plt.show()
g = sns.countplot(train_labels)
# normalisation of the data so its between 0 and 1

train_features = train_features.astype('float32')/255.

test_features = test_features.astype('float32')/255.
# convert the target labels to categorical 

train_labels = to_categorical(train_labels)

train_features.shape, train_labels.shape
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.10, random_state=42)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
# setup the basic CNN

model = Sequential()

model.add(layers.Conv2D(32, kernel_size=(3, 3), 

                    activation='relu', 

                    kernel_initializer='he_normal',

                    input_shape=(28, 28, 1)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.20))



model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.25))



model.add(layers.Conv2D(128, (3, 3), activation='relu'))

#model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.3))

          

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dropout(0.4))

model.add(layers.Dense(10, activation='softmax'))
# Add optimser and compile the model.

optimiser = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer = optimiser ,metrics=['accuracy'])
model.summary()
# Model Parameters with learning rate scheduler

batchsize = 32

num_classes = 10

epochs = 40

stepsperepoch = X_train.shape[0]//64

annealer = LearningRateScheduler(lambda x: 1e-4 * 0.9 ** x)
# Data augmentation for the images

# datagen = ImageDataGenerator(

#                     rotation_range=25,  

#                     zoom_range = 0.10,  

#                     width_shift_range=0.2, 

#                     height_shift_range=0.2)

from keras.utils import Sequence

datagen = ImageDataGenerator(rotation_range=25,

                                zoom_range = 0.10,  

                                width_shift_range=0.2,

                                height_shift_range=0.2)



train_batches = datagen.flow(X_train, y_train, batch_size=32)
# fit the model 

history = model.fit_generator(generator=train_batches, 

                    steps_per_epoch = stepsperepoch,

                    epochs=epochs,

                    verbose=1,

                    validation_data=(X_val, y_val),

                    callbacks=[annealer])
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_val, y_val, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

history_dict = history.history
# Plot Loss

loss = history_dict['loss']

val_loss = history_dict['val_loss']



epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, color='b', label='Training loss')

plt.plot(epochs, val_loss, color='r', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
# Plot Accuracy

acc = history_dict['accuracy']

val_acc = history_dict['val_accuracy']



plt.clf() 

plt.plot(epochs, acc, color='b', label='Training acc')

plt.plot(epochs, val_acc, color='r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
predict = model.predict_classes(test_features, verbose=0)

predict.shape
result=pd.DataFrame({'ImageId': list(range(1,len(predict)+1)),

                         'Label': predict})
result.to_csv('DSKsubmission.csv', index=False)
result