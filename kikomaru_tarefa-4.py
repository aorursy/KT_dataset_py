import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
lb = pd.read_csv('../input/train_labels.csv')
lb.groupby(['label']).size()
a = np.load('../input/train_images_pure.npy')

plt.subplot(2,2,1)
plt.imshow(a[0], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,2)
plt.imshow(a[1], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,3)
plt.imshow(a[2], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,4)
plt.imshow(a[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
a = np.load('../input/train_images_noisy.npy')

plt.subplot(2,2,1)
plt.imshow(a[0], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,2)
plt.imshow(a[1], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,3)
plt.imshow(a[2], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,4)
plt.imshow(a[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
a = np.load('../input/train_images_rotated.npy')

plt.subplot(2,2,1)
plt.imshow(a[0], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,2)
plt.imshow(a[1], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,3)
plt.imshow(a[2], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,4)
plt.imshow(a[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
a = np.load('../input/train_images_both.npy')

plt.subplot(2,2,1)
plt.imshow(a[0], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,2)
plt.imshow(a[1], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,3)
plt.imshow(a[2], cmap=plt.get_cmap('gray'))
plt.subplot(2,2,4)
plt.imshow(a[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th') #formato 'channels first'
def baseline_model(KERNEL_SIZE, NUM_CLASS, INPUT_SHAPE):
    model = Sequential()
    
    model.add(Conv2D(20, kernel_size=KERNEL_SIZE,strides=(2,2), padding='same', input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    
    model.add(Conv2D(50, kernel_size=KERNEL_SIZE, strides=(2,2), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    
    model.add(Dense(NUM_CLASS))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
NUM_CLASS = 10
INPUT_SHAPE = (1, 28, 28)
NUM_EPOCHS = 20
BATCH_SIZE = 128
VERBOSE = 1
KERNEL_SIZE = (3,3)
baseline_model_1 = baseline_model(KERNEL_SIZE, NUM_CLASS, INPUT_SHAPE)
baseline_model_1.summary()
X_train = np.load('../input/train_images_pure.npy')
y_train = pd.read_csv('../input/train_labels.csv')
X_train.shape
y_train = y_train.drop('Id', axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_val = X_val.reshape(X_val.shape[0], 1, 28, 28).astype('float32')
y_train = np_utils.to_categorical(y_train, NUM_CLASS)
y_val = np_utils.to_categorical(y_val, NUM_CLASS)
print(X_train.shape)
print(y_train.shape)
print()
print(X_val.shape)
print(y_val.shape)
from keras.callbacks import EarlyStopping
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
history = baseline_model_1.fit(X_train, y_train,validation_data=(X_val, y_val), epochs = NUM_EPOCHS,
                    batch_size=BATCH_SIZE, verbose=VERBOSE,callbacks = callbacks)
X_test = np.load('../input/train_images_rotated.npy')
y_test = pd.read_csv('../input/train_labels.csv')
y_test = y_test.drop('Id', axis=1)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
y_test = np_utils.to_categorical(y_test, NUM_CLASS)
scores = baseline_model_1.evaluate(X_test, y_test, verbose=0)
scores
print('CNN Error:', (100 - 100*scores[1]), '%')
KERNEL_SIZE = (5,5)
baseline_model_2 = baseline_model(KERNEL_SIZE, NUM_CLASS, INPUT_SHAPE)
history = baseline_model_2.fit(X_train, y_train,validation_data=(X_val, y_val), epochs = NUM_EPOCHS,
                    batch_size=BATCH_SIZE, verbose=VERBOSE,callbacks = callbacks)
scores = baseline_model_2.evaluate(X_test, y_test, verbose=0)
print('CNN Error:', (100 - 100*scores[1]), '%')
KERNEL_SIZE = (2,2)
baseline_model_3 = baseline_model(KERNEL_SIZE, NUM_CLASS, INPUT_SHAPE)
history = baseline_model_3.fit(X_train, y_train,validation_data=(X_val, y_val), epochs = NUM_EPOCHS,
                    batch_size=BATCH_SIZE, verbose=VERBOSE,callbacks = callbacks)
scores = baseline_model_3.evaluate(X_test, y_test, verbose=0)
print('CNN Error:', (100 - 100*scores[1]), '%')
def baseline_model():
    model = Sequential()
    
    #model.add(Conv2D(20, kernel_size=(2,2), strides=(2,2), padding='same', input_shape=(1, 28, 28)))
    #model.add(Activation('relu'))
    
    model.add(Conv2D(50, kernel_size=(2,2), strides=(2,2), padding='same', input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
baseline_model_5 = baseline_model()
history = baseline_model_5.fit(X_train, y_train,validation_data=(X_val, y_val), epochs = NUM_EPOCHS,
                    batch_size=BATCH_SIZE, verbose=VERBOSE,callbacks = callbacks)
baseline_model_5.summary()
scores = baseline_model_5.evaluate(X_test, y_test, verbose=0)
print('CNN Error:', (100 - 100*scores[1]), '%')
def baseline_model():
    model = Sequential()
    
    model.add(Conv2D(20, kernel_size=(2,2), padding='same', input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(50, kernel_size=(2,2), padding='same'))
    model.add(Activation('relu'))
    
    model.add(Conv2D(10, kernel_size=(2,2), padding='same'))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
baseline_model_6 = baseline_model()
baseline_model_6.summary()
history = baseline_model_6.fit(X_train, y_train,validation_data=(X_val, y_val), epochs = NUM_EPOCHS,
                    batch_size=BATCH_SIZE, verbose=VERBOSE,callbacks = callbacks)
scores = baseline_model_6.evaluate(X_test, y_test, verbose=0)
print('CNN Error:', (100 - 100*scores[1]), '%')
def baseline_model():
    model = Sequential()
    
    model.add(Conv2D(20, kernel_size=(2,2), padding='same', input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(50, kernel_size=(2,2), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    
    #model.add(Dense(128))
    #model.add(Activation('relu'))
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
baseline_model_7 = baseline_model()
baseline_model_7.summary()
history = baseline_model_7.fit(X_train, y_train,validation_data=(X_val, y_val), epochs = NUM_EPOCHS,
                    batch_size=BATCH_SIZE, verbose=VERBOSE,callbacks = callbacks)
scores = baseline_model_7.evaluate(X_test, y_test, verbose=0)
print('CNN Error:', (100 - 100*scores[1]), '%')
def baseline_model():
    model = Sequential()
    
    model.add(Conv2D(20, kernel_size=(2,2), padding='same', input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(50, kernel_size=(2,2), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
baseline_model_8 = baseline_model()
baseline_model_8.summary()
history = baseline_model_8.fit(X_train, y_train,validation_data=(X_val, y_val), epochs = NUM_EPOCHS,
                    batch_size=BATCH_SIZE, verbose=VERBOSE,callbacks = callbacks)
scores = baseline_model_8.evaluate(X_test, y_test, verbose=0)
print('CNN Error:', (100 - 100*scores[1]), '%')
X_test_2 = np.load('../input/train_images_both.npy') #Teste nos dados com as duas interferências
X_test_2 = X_test_2.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
X_test_2 /= 255
scores = baseline_model_3.evaluate(X_test_2, y_test, verbose=0) #O 3o modelo foi o melhor.
print('CNN Error:', (100 - 100*scores[1]), '%')
def pool_model():
    model = Sequential()
    
    model.add(Conv2D(20, kernel_size=(2,2), padding='same', input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(50, kernel_size=(2,2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
pool_model = pool_model()
pool_model.summary()
history = pool_model.fit(X_train, y_train,validation_data=(X_val, y_val), epochs = NUM_EPOCHS,
                    batch_size=BATCH_SIZE, verbose=VERBOSE,callbacks = callbacks)
scores = pool_model.evaluate(X_test, y_test, verbose=0)
print('CNN Error:', (100 - 100*scores[1]), '%')
scores = pool_model.evaluate(X_test_2, y_test, verbose=0)
print('CNN Error:', (100 - 100*scores[1]), '%')
test = np.load('../input/Test_images.npy')
test = test.reshape(test.shape[0], 1, 28, 28).astype('float32')
test /= 255
a = pd.read_csv('../input/sample_sub.csv')
a
predict = pool_model.predict_classes(test)
predict
prediction = pd.DataFrame({'Id':a['Id'], 'label':predict})
prediction.to_csv('submition.csv', index=False)