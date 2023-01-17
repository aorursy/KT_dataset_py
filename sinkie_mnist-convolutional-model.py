import keras; print(keras.__version__)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
path = '../input/'

%ls $path
train = pd.read_csv(path+'train.csv')

test = pd.read_csv(path+'test.csv')

train.head()
# Create the training features (without labels)

X = train.drop('label',axis=1).values



# Create a data normalizer

mean_x = X.mean().astype(np.float32)

std_x = X.std().astype(np.float32)

def process_input(X):

    length = X.shape[0]

    return ((X - mean_x)/std_x).reshape(length,28,28,1)



# Normalize training and test data

X = process_input(X)

test_X = process_input(test.values)



# One hot encode the target labels using pandas get_dummies

y = pd.get_dummies(train.label).values



print("X shape: {}".format(X.shape))

print("y shape: {}".format(y.shape))

print("y:",y[:5,:])

# print("X:",X[0,:,:,:])
np.random.seed(999)

mask = np.random.rand(X.shape[0]) < 0.9

X_train = X[mask]

X_val = X[~mask]

y_train = y[mask]

y_val = y[~mask]

print(X_train.shape,y_train.shape,X_val.shape,y_val.shape)
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Reshape, Flatten, Lambda

from keras.optimizers import SGD, Adam

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPooling2D

from keras.layers.normalization import BatchNormalization



# Define sequential model

model = Sequential()

model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', activation='relu',input_shape=(28,28,1)))

model.add(BatchNormalization(axis=1))

model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', activation='relu'))

model.add(BatchNormalization(axis=1))

model.add(Conv2D(256, (3,3), strides=(2,2), padding='same', activation='relu'))

model.add(BatchNormalization(axis=1))

model.add(Flatten())

model.add(Dense(8, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))



# Compile model

model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])



# Display a summary of the model

model.summary()
# Fit model

model.optimizer.lr = 0.001

hist = model.fit(X, y, batch_size=64, epochs=1, verbose=1, callbacks=None, validation_split=0.2, shuffle=True)



# In reality train for a few more epochs

model.optimizer.lr = 0.01

hist2 = model.fit(X, y, batch_size=64, epochs=2, verbose=1, callbacks=None, validation_split=0.2, shuffle=True)

model.optimizer.lr = 0.001

hist3 = model.fit(X, y, batch_size=64, epochs=2, verbose=1, callbacks=None, validation_split=0.2, shuffle=True)
# Save model weights for future use / improvement

model.save_weights('MNIST_tf.h5')
import matplotlib.pyplot as plt



# Combine all three histories to be plotted on one graph

hist_all_acc = np.concatenate([hist.history['acc'],hist2.history['acc'],

                               hist3.history['acc']])

hist_all_val_acc = np.concatenate([hist.history['val_acc'],hist2.history['val_acc'],

                                   hist3.history['val_acc']])

# Plot data

plt.plot(hist_all_acc)

plt.plot(hist_all_val_acc)

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
pred = model.predict(test_X)

preds = np.argmax(pred,axis=1)

sub = {'ImageId':test.index +1,'Label':preds}

sub = pd.DataFrame(sub)

sub.to_csv('MNIST_submission.csv',index=False)

sub.head()
from keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(channel_shift_range=2,

                         rotation_range=15,

                         width_shift_range=0.1,

                         height_shift_range=0.15)



batch_size = 10

batch = next(gen.flow(X,y,batch_size=batch_size))

imgs = batch[0]

labels = batch[1]



#Plot a sample of generated images

plt.figure(figsize=(20,10))

columns = 5

for i, image in enumerate(imgs):

    plt.subplot(batch_size / columns + 1, columns, i + 1)

    plt.imshow(image.reshape(28,28))
#Generate large amount of additional data and append to real data

batch_size = 128

X_aug = np.copy(X_train)

y_aug = np.copy(y_train)

for i in range(2): #make this e.g. 500 to generate a large amount of additional data

    if i%10==0: print(i)

    batch = next(gen.flow(X,y,batch_size=batch_size))

    X_aug = np.concatenate([X_aug,batch[0]])

    y_aug = np.concatenate([y_aug,batch[1]])



print(X_train.shape,y_train.shape)

print(X_aug.shape,y_aug.shape)
# Fine tune model

model.optimizer.lr = 0.01

hist = model.fit(X_aug, y_aug, batch_size=64, epochs=1, verbose=1, callbacks=None,

                 validation_data=(X_val,y_val), shuffle=True)



# Train for further epochs...
# Plot learning curve

plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
pred = model.predict(test_X)

preds = np.argmax(pred,axis=1)

sub = {'ImageId':test.index +1,'Label':preds}

sub = pd.DataFrame(sub)

sub.to_csv('MNIST_submission_2.csv',index=False)

sub.head()