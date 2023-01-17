import numpy as np

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.callbacks import LearningRateScheduler



import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load data from csv files

train_df = pd.read_csv('../input/Kannada-MNIST/train.csv')

test_df = pd.read_csv('../input/Kannada-MNIST/test.csv')

dig_df = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')

print(train_df.shape, test_df.shape, dig_df.shape)
train_df['label'].value_counts().sort_index()
dig_df['label'].value_counts().sort_index()
# create arrays from dataframes

X = train_df.drop(['label'], axis=1).values

Y = train_df['label']

X_test = test_df.drop(['id'], axis=1).values

X_dig = dig_df.drop(['label'], axis=1).values

Y_dig = dig_df['label']

print(X.shape, Y.shape, X_test.shape)

print(X_dig.shape, Y_dig.shape)
import matplotlib.pyplot as plt



# look at some of the digits from train dataset

plt.figure(figsize=(15,6))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(X[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.title("label=%d" % Y[i],y=0.9)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
# look at some of the digits from dig dataset

plt.figure(figsize=(15,6))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(X_dig[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.title("label=%d" % Y_dig[i],y=0.9)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
# look at some of the digits from test dataset

plt.figure(figsize=(15,6))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(X_test[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
# set up a dev set to check the performance of the CNN classifier

from sklearn.model_selection import train_test_split

X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.1)
print(X_train.shape, Y_train.shape)

print(X_dev.shape, Y_dev.shape)
# prepare the data for CNN



# reshape flattened data into 3D tensor & standardize the values in the datasets by dividing by 255

n_x = 28

train_img = X_train.reshape((-1, n_x, n_x, 1)).astype('float32')/255.

dev_img = X_dev.reshape((-1, n_x, n_x, 1)).astype('float32')/255.     # similarly for dev set

test_img = X_test.reshape((-1, n_x, n_x, 1)).astype('float32')/255.   # similarly for test set

dig_img = X_dig.reshape((-1, n_x, n_x, 1)).astype('float32')/255.     # similarly for dig set

print(train_img.shape, dev_img.shape, test_img.shape, dig_img.shape)



# one-hot encode the labels in Y_train, Y_dev, Y_dig

from keras.utils.np_utils import to_categorical

train_labels = to_categorical(Y_train)

dev_labels = to_categorical(Y_dev)

dig_labels = to_categorical(Y_dig)

print(train_labels.shape, dev_labels.shape, dig_labels.shape)

print(Y_dig[8], dig_labels[8])

plt.figure(figsize=(1,1))

plt.imshow(X_dig[8].reshape((28,28)),cmap=plt.cm.binary)

plt.show()
# use Keras data generator to augment the training set



from keras_preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint



data_augment = ImageDataGenerator(rescale = 1./255, width_shift_range=0.1, height_shift_range=0.1)



checkpoint = ModelCheckpoint('BWeight.md5', monitor='val_loss', save_best_only=True)

# build the CNN from keras

from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))

model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))

model.add(layers.BatchNormalization(momentum=0.15))

model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))

model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))

model.add(layers.BatchNormalization(momentum=0.15))

model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(64, kernel_size=5, padding='same', activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))

model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))

model.add(layers.BatchNormalization(momentum=0.15))

model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(128, kernel_size=5, padding='same', activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation='softmax'))

model.summary()
from keras.optimizers import RMSprop

from tensorflow.keras.callbacks import EarlyStopping



es = EarlyStopping(monitor='val_loss', verbose=2, patience=10)



initial_learningrate=2e-3





# compile the model

#model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

model.compile(optimizer=RMSprop(lr=initial_learningrate), loss='categorical_crossentropy',  metrics=['accuracy'])
# set a learning rate annealer

from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',patience=10,factor=0.6,min_lr=0.00001, verbose=2)





def lr_decay(epoch):#lrv

    return initial_learningrate * 0.99 ** epoch
# Train and validate the model

epochs = 50

batch_size = 64

history = model.fit_generator(data_augment.flow(train_img, train_labels, batch_size=batch_size), 

                              epochs=epochs, steps_per_epoch=train_img.shape[0]//batch_size, 

                              validation_data=(dev_img, dev_labels), callbacks=[learning_rate_reduction,LearningRateScheduler(lr_decay),es,checkpoint])



model.load_weights('BWeight.md5')
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
model.evaluate(dig_img, dig_labels)
pred_dig = model.predict(dig_img)

dig_df['pred'] = np.argmax(pred_dig, axis=1)
# look at those that were classified wrongly in X_dig

dig_df['correct'] = dig_df['label'] - dig_df['pred']

errors = dig_df[dig_df['correct'] != 0]

error_list = errors.index

print('Number of errors is ', len(errors))

print('The indices are ', error_list)
# plot images of some of the wrong predictions for X_dig

plt.figure(figsize=(15,10))

for i in range(60):

    plt.subplot(6, 10, i+1)

    plt.imshow(X_dig[error_list[i]].reshape((28,28)),cmap=plt.cm.binary)

    plt.title("true={}\npredict={}".format(dig_df['label'][error_list[i]], 

                                           dig_df['pred'][error_list[i]]), y=0.9)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
# predict on test set

predictions = model.predict(test_img)

print(predictions.shape)
# set the predicted labels to be the one with the highest probability

predicted_labels = np.argmax(predictions, axis=1)
# look at some of the predictions for test_X

plt.figure(figsize=(15,6))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(test_img[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.title("predict=%d" % predicted_labels[i],y=0.9)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
# create submission file

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = predicted_labels

# generate submission file in csv format

submission.to_csv('submission.csv', index=False)
submission