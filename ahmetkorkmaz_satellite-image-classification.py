#Here are some standard libraries that are loaded when you 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualize satellite images

from skimage.io import imshow # visualize satellite images

from sklearn.model_selection import train_test_split

from keras.utils import np_utils

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout # components of network

from keras.models import Sequential # type of model

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import BatchNormalization

from keras.callbacks import LearningRateScheduler

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras.models import load_model
x_train_set_fpath = '../input/X_test_sat4.csv'

y_train_set_fpath = '../input/y_test_sat4.csv'

print ('Loading Training Data')

X_train = pd.read_csv(x_train_set_fpath)

print ('Loaded 28 x 28 x 4 images')

y_train = pd.read_csv(y_train_set_fpath)

print ('Loaded labels')

X_train= X_train.values.reshape([-1,28,28,4])

X_train = X_train / 255.0

Y_train = y_train

print(X_train.shape)

print(Y_train.shape)
X_train_img = X_train.reshape([99999,28,28,4]).astype(float)

print (X_train_img.shape)

ix = 5

imshow(np.squeeze(X_train_img[ix,:,:,0:3]).astype(float)) #Only seeing the RGB channels

plt.show()
epochs = 50

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=0, save_best_only=True)

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.2)
model = Sequential()

model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 4)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(128, kernel_size = 4, activation='relu'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dropout(0.4))

model.add(Dense(4, activation='softmax'))

    

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#model.summary()

history=model.fit(X_train2,Y_train2,batch_size=64, epochs=epochs,callbacks=[annealer,es,mc] ,verbose=0, validation_data = (X_val2,Y_val2) )

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
saved_model = load_model('best_model.h5')

# evaluate the model

_, train_acc = saved_model.evaluate(X_train2,Y_train2, verbose=0)

_, test_acc = saved_model.evaluate(X_val2,Y_val2, verbose=0)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

preds = saved_model.predict(X_train[-1000:], verbose=1)
ix = 12

imshow(np.squeeze(X_train_img[ix,:,:,0:3]).astype(float)) #Only seeing the RGB channels



plt.show()

#Tells what the image is

print ('Prediction:\n{:.1f}% probability barren land,\n{:.1f}% probability trees,\n{:.1f}% probability grassland,\n{:.1f}% probability other\n'.format(preds[ix,0]*100,preds[ix,1]*100,preds[ix,2]*100,preds[ix,3]*100))