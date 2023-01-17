# importing essentials

import numpy as np

import pandas as pd



# to make sure we use tensorflow as our backend, so the image format will be depth/channel last

from keras import backend as K

K.set_image_dim_ordering('tf')

import matplotlib.pyplot as plt

%matplotlib inline
# load training data

train_data = pd.read_csv('../input/train.csv')
# check the data shape

train_data.shape
# separate the training data into images and labels

images = train_data.iloc[:, 1:]

labels = train_data.iloc[:, 0]
# reshape the images so we can use CNNs

# we are following tensorflow image format

labels = labels.as_matrix()

images = images.as_matrix().reshape(images.shape[0], 28, 28, 1)
# features normalisation

def normalize_grayscale(image_data):

    # Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]

    # you may also want to try to scale to [-1, 1] or just use keras Batchnormalisation layer

    return (25.5 + 0.8 * image_data) / 255

train_features = normalize_grayscale(images)
# one-hot encoding for labels

from keras.utils import np_utils

train_labels = np_utils.to_categorical(labels)
# divide data into training and validation set

from sklearn.model_selection import train_test_split

train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, 

                                                                           test_size=0.15, random_state=np.random.randint(300))
# check if anything wrong

print('train_features shape: ', train_features.shape)

print('val_features shape: ', val_features.shape)

print('train_labels shape: ', train_labels.shape)

print('val_labels shape: ', val_labels.shape)
# hyperparameters

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout

from keras.models import Model, load_model





# build model





# model = Sequential()

# model.add(Conv2D(64, (5, 5), input_shape=(1, 28, 28), activation='relu',

#           bias_initializer='RandomNormal'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(128, (5, 5), activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())

# model.add(Dense(256, activation='relu'))

# model.add(Dropout(0.2))

# model.add(Dense(32, activation='relu'))

# model.add(Dropout(0.2))

# model.add(Dense(n_classes, activation='softmax'))



'''let's follow the fashion of creating model using keras'''

def get_model(input_shape):

    

    drop = 0.3

    

    X_input = Input(input_shape)

    

    X = Conv2D(64, (5,5), strides=(1,1), activation='relu', 

               kernel_initializer='glorot_normal')(X_input)

    X = MaxPooling2D((2,2))(X)

    

    X = Conv2D(128, (5,5), strides=(1,1), activation='relu',

              kernel_initializer='glorot_normal')(X)

    

    X = MaxPooling2D((2,2))(X)

    

    X = Flatten()(X)

    

    X = Dense(256, activation='relu')(X)

    X = Dropout(drop)(X)

    

    X = Dense(32, activation='relu')(X)

    X = Dropout(drop)(X)

    

    X = Dense(10, activation='softmax')(X)

    

    model = Model(inputs=[X_input], outputs=[X])

    

    return model

# optimizer, the learning rate will decay with time by default settings

# Adam is a comman practice

from keras.optimizers import Nadam

opt = Nadam(lr=0.001)
model = get_model((28, 28, 1))

# compile the model

model.compile(optimizer=opt,

              loss='categorical_crossentropy',

              metrics=['accuracy'])
# model summary

model.summary()
from keras.callbacks import ModelCheckpoint

# only save the best model

f_path = 'model.h5'

msave = ModelCheckpoint(f_path, save_best_only=True)
# training

epochs = 5

batch_size = 64

training = model.fit(train_features, train_labels,

                     validation_data=(val_features, val_labels),

                     epochs=epochs,

                     callbacks=[msave],

                     batch_size=batch_size, 

                     verbose=1)

# show the loss and accuracy

loss = training.history['loss']

val_loss = training.history['val_loss']

acc = training.history['acc']

val_acc = training.history['val_acc']



# loss plot

tra = plt.plot(loss)

val = plt.plot(val_loss, 'r')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Loss')

plt.legend(["Training", "Validation"])



plt.show()



# accuracy plot

plt.plot(acc)

plt.plot(val_acc, 'r')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Accuracy')

plt.legend(['Training', 'Validation'], loc=4)

plt.show()
# load the best model

model = load_model(f_path)



# load test_data

test_data = pd.read_csv('../input/test.csv')



# reshape the test_data

test_images = test_data.as_matrix().reshape(test_data.shape[0], 28, 28, 1)



# normalisation

test_features = normalize_grayscale(test_images)



# prediction

pred = model.predict(test_features, batch_size=batch_size, 

                       verbose=1)



# convert predicions from categorical back to 0...9 digits

pred_digits = np.argmax(pred, axis=1)



submission = pd.DataFrame({'Label': pred_digits})

submission.index += 1

submission.index.name = "ImageId"

submission.to_csv('submission.csv')