import numpy as np

import pandas as pd

import os



from sklearn.model_selection import train_test_split

from scipy.signal import convolve2d



import matplotlib.pyplot as plt

%matplotlib inline



from skimage.color import rgb2gray

from skimage.transform import resize



import tensorflow as tf



from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tensorflow.keras import utils

from tensorflow.keras.applications.inception_v3 import InceptionV3



# se imports

import time

from sklearn.dummy import DummyClassifier
plt.rcParams['font.size'] = 16
# define a simple CNN model

def build_mnist_CNN():

    mnist_model = Sequential()

    mnist_model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))

    mnist_model.add(MaxPooling2D(pool_size=(2, 2)))

    mnist_model.add(Dropout(0.2))

    mnist_model.add(Flatten())

    mnist_model.add(Dense(128, activation='relu'))

    mnist_model.add(Dense(num_classes, activation='softmax'))



    # Compile model

    mnist_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return mnist_model
# load data

(X_train, y_train), (X_test, y_test) = mnist.load_data()



# reshape to be [samples][channels][width][height]

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')



# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

X_test = X_test / 255



# one hot encode outputs

y_train = utils.to_categorical(y_train)

y_test = utils.to_categorical(y_test)

num_classes = y_test.shape[1]



# take a subset of the data for speed

subset_size = 10000

X_train = X_train[:subset_size]

y_train = y_train[:subset_size]



mnist_model = build_mnist_CNN()



# Fit the model

start_time = time.time()

mnist_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=256)

# Final evaluation of the model

scores = mnist_model.evaluate(X_test, y_test, verbose=0)

print("CNN Error: %.2f%%" % (100-scores[1]*100))

elapsed_time = time.time()-start_time

print("---Running Time: %s seconds ---" % elapsed_time)



mnist_model.summary()
# load data

(X_train, y_train), (X_test, y_test) = mnist.load_data()



# reshape to be [samples][channels][width][height]

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')



# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

X_test = X_test / 255



# one hot encode outputs

y_train = utils.to_categorical(y_train)

y_test = utils.to_categorical(y_test)

num_classes = y_test.shape[1]



# take a subset of the data for speed

# subset_size = 10000

# X_train = X_train[:subset_size]

# y_train = y_train[:subset_size]



mnist_model = build_mnist_CNN()



# Fit the model

start_time = time.time()

mnist_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=256)

# Final evaluation of the model

scores = mnist_model.evaluate(X_test, y_test, verbose=0)

print("CNN Error: %.2f%%" % (100-scores[1]*100))

elapsed_time = time.time()-start_time

print("---Running Time: %s seconds ---" % elapsed_time)



mnist_model.summary()
data = pd.read_csv('../input/dog-breed-identification/labels.csv')

data = data[:2000]

data['image_path'] = data.apply( lambda row: (os.path.join("../input/dog-breed-identification/train", row["id"] + ".jpg") ), axis=1)

data.head()
target_labels = data['breed']

total_classes = len(set(target_labels))

print("number of dog breeds:", total_classes)
# read images from the image directory. 

images = np.array([img_to_array(

                    load_img(img, target_size=(256,256))

                    ) for img in data['image_path'].values.tolist()])
images.shape
images = images.astype('float32')/255.0 
plt.imshow(images[0]);

plt.grid(True);

plt.xticks([]);

plt.yticks([]);

plt.title("Breed = " + target_labels[0]);
X_train, X_valid, y_train, y_valid = train_test_split(images, target_labels, 

                                                    stratify=np.array(target_labels), 

                                                    random_state=42)



print(X_train.shape)

print(X_valid.shape)
y_train.shape
y_train.value_counts()
y_train.value_counts().hist()
1 - 20/1500
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

print(f"Train error: {1 - dummy.score(X_train, y_train)}")

print(f"Valid error: {1 - dummy.score(X_valid, y_valid)}")
# OHE



Y_train = pd.get_dummies(y_train.reset_index(drop=True)).values

Y_valid = pd.get_dummies(y_valid.reset_index(drop=True)).values



print(Y_train.shape)

print(Y_valid.shape)



# Note: it would be better to use keras.utils.to_categorical, or something else like that,

# just in case one of the classes is absent in one of the two sets.

# But this works for now.
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))

model.add(Activation('relu')) # this is just different syntax for specifying the activation function

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(total_classes))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()



history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_valid, Y_valid))



# FYI: it's often a good idea to save your weights after training or during training.

# But you don't have to here.

# model.save_weights('my_conv_net.h5')  
model.evaluate(X_valid, Y_valid, verbose=0)
# Get the InceptionV3 model trained on the ImageNet data set

base_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))



top_block = base_inception.output

top_block = GlobalAveragePooling2D()(top_block) # pool over height/width to reduce number of parameters

top_block = Dense(256, activation='relu')(top_block) # add a Dense layer

predictions = Dense(total_classes, activation='softmax')(top_block) # add another Dense layer



model_transfer = Model(inputs=base_inception.input, outputs=predictions)



for layer in base_inception.layers:

    layer.trainable = False

    

model_transfer.compile(Adam(lr=.001), loss='categorical_crossentropy', metrics=['accuracy']) 

model_transfer.summary() # run me if you dare

history = model_transfer.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=10)
model_transfer.evaluate(X_valid, Y_valid, verbose=0)
for i, layer in enumerate(reversed(model_transfer.layers)):

    layer.trainable = True

#     print(layer)

    if i > 15:

        break



# compile the model with a SGD/momentum optimizer and a very slow learning rate.

model_transfer.compile(loss='categorical_crossentropy',

              optimizer=SGD(lr=1e-4, momentum=0.9),

              metrics=['accuracy'])



# fine-tune the unfrozen layers

history = model_transfer.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=10)
model_transfer.evaluate(X_valid, Y_valid, verbose=0)