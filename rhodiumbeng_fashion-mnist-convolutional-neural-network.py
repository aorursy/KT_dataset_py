import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))
# load data

train_df = pd.read_csv('../input/fashion-mnist_train.csv')

test_df = pd.read_csv('../input/fashion-mnist_test.csv')

print(train_df.shape, test_df.shape)
# create dict for the class labels

index_to_class = {0:'T-shirt/top',

                  1:'Trouser',

                  2:'Pullover',

                  3:'Dress',

                  4:'Coat',

                  5:'Sandal',

                  6:'Shirt',

                  7:'Sneaker',

                  8:'Bag',

                  9:'Ankle boot'}
# create arrays from dataframes

train_X = train_df.drop(['label'], axis=1).values

train_Y = train_df['label'].values

test_X = test_df.drop(['label'], axis=1).values

test_Y = test_df['label'].values

print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
import matplotlib.pyplot as plt



# look at some of the pics from train_X

plt.figure(figsize=(15,10))

for i in range(40):  

    plt.subplot(5, 8, i+1)

    plt.imshow(train_X[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.title("label=%s" % index_to_class[train_Y[i]],y=1)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
# prepare the data for CNN



# reshape flattened data into 3D tensor

n_x = 28

train_X_pic = train_X.reshape((-1, n_x, n_x, 1))  

test_X_pic = test_X.reshape((-1, n_x, n_x, 1))    # similarly for test set

print(train_X_pic.shape, test_X_pic.shape)



# standardize the values in the datasets by dividing by 255

train_X_pic = train_X_pic / 255.

test_X_pic = test_X_pic / 255.
# one-hot encode the labels in train_Y

from keras.utils.np_utils import to_categorical

train_labels = to_categorical(train_Y)

print(train_labels.shape)

print(train_Y[181], index_to_class[train_Y[181]], train_labels[181])

plt.figure(figsize=(1,1))

plt.imshow(train_X[181].reshape((28,28)),cmap=plt.cm.binary)

plt.show()
# one-hot encode the labels in test_Y

test_labels = to_categorical(test_Y)

print(test_labels.shape)

print(test_Y[181], index_to_class[test_Y[181]], test_labels[181])

plt.figure(figsize=(1,1))

plt.imshow(test_X[181].reshape((28,28)),cmap=plt.cm.binary)

plt.show()
# use Keras data generator to augment the training set



from keras_preprocessing.image import ImageDataGenerator

data_augment = ImageDataGenerator(rotation_range=10,

                                 width_shift_range=0.1,

                                 height_shift_range=0.1,

                                 zoom_range=0.1)
# build the CNN from keras

from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))

model.add(layers.Conv2D(32, kernel_size=3, padding='valid', activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))

model.add(layers.Dropout(rate=0.4))

model.add(layers.Conv2D(64, kernel_size=5, padding='same', activation='relu'))

model.add(layers.Conv2D(64, kernel_size=5, padding='valid', activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))

model.add(layers.Dropout(rate=0.4))

model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))

model.add(layers.Conv2D(128, kernel_size=3, padding='valid', activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))



model.summary()
# compile the model

model.compile(optimizer='adam', loss='categorical_crossentropy', 

              metrics=['accuracy'])
# set up a dev set (5000 samples) to check the performance of the CNN

X_dev = train_X_pic[:5000]

rem_X_train = train_X_pic[5000:]

print(X_dev.shape, rem_X_train.shape)



Y_dev = train_labels[:5000]

rem_Y_train = train_labels[5000:]

print(Y_dev.shape, rem_Y_train.shape)
# Train and validate the model

epochs = 100

batch_size = 128

history = model.fit_generator(data_augment.flow(rem_X_train, rem_Y_train, batch_size=batch_size), 

                              epochs=epochs, steps_per_epoch=rem_X_train.shape[0]//batch_size, 

                              validation_data=(X_dev, Y_dev))
# plot and visualise the training and validation losses

loss = history.history['loss']

dev_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)



from matplotlib import pyplot as plt

plt.plot(epochs, loss, 'bo', label='training loss')

plt.plot(epochs, dev_loss, 'b', label='validation loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
# do error analysis on the predictions for X_dev

pred_dev = model.predict(X_dev)

pred_dev_labels = []

for i in range(5000):

    pred_dev_label = np.argmax(pred_dev[i])

    pred_dev_labels.append(pred_dev_label)
# look at those that were classified wrongly in X_dev

result = pd.DataFrame(train_Y[:5000], columns=['Y_dev'])

result['Y_pred'] = pred_dev_labels

result['correct'] = result['Y_dev'] - result['Y_pred']

errors = result[result['correct'] != 0]

error_list = errors.index

print('Number of errors is ', len(errors))

print('The indices are ', error_list)
# plot the image of the wrong in predictions for X_dev

plt.figure(figsize=(15,90))

for i in range(len(error_list)):

    plt.subplot(43, 8, i+1)

    plt.imshow(X_dev[error_list[i]].reshape((28,28)),cmap=plt.cm.binary)

    plt.title("true={}\npredict={}".format(index_to_class[train_Y[error_list[i]]], 

                                           index_to_class[pred_dev_labels[error_list[i]]]), y=1)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
# predict on test set

predictions = model.predict(test_X_pic)

print(predictions.shape)
model.evaluate(x=test_X_pic, y=test_labels)
# set the predicted labels to be the one with the highest probability

predicted_labels = []

for i in range(10000):

    predicted_label = np.argmax(predictions[i])

    predicted_labels.append(predicted_label)
# look at some of the predictions for test_X

plt.figure(figsize=(15,12))

for i in range(40):  

    plt.subplot(5, 8, i+1)

    plt.imshow(test_X[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.title("true={}\npredict={}".format(index_to_class[test_Y[i]], 

                                           index_to_class[predicted_labels[i]]), y=1)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()