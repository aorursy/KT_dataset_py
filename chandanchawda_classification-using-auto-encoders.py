import keras

import numpy as np

import os

import matplotlib.pylab as plt

from imgaug import augmenters

from keras.layers import Dense, Input, Conv2D, LSTM, MaxPool2D, UpSampling2D, Flatten

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from numpy import argmax, array_equal

import matplotlib.pyplot as plt

from keras.models import Model

from imgaug import augmenters
# from keras.datasets import mnist



def load_mnist_dataset(path):

    with np.load(path) as f:

        x_train, y_train = f['x_train'], f['y_train']

        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)

    

(train_data, train_labels), (test_data, test_labels) = load_mnist_dataset('../input/mnist.npz')

train_x, val_x = train_test_split(train_data, test_size=0.2)



train_x = train_x/255.

val_x = val_x/255.

train_x = train_x.reshape(-1, 28, 28, 1)

val_x = val_x.reshape(-1, 28, 28, 1)



gaussian_noise = augmenters.GaussianBlur(sigma=(0, 3.0))

sequential_object = augmenters.Sequential([gaussian_noise])



train_x_noisy = sequential_object.augment_images(train_x * 255) / 255

val_x_noisy = sequential_object.augment_images(val_x * 255) / 255
f, ax = plt.subplots(1,10)

for i in range(0,10):

    ax[i].imshow(train_x[i].reshape(28, 28))

plt.show()
f, ax = plt.subplots(1,10)

for i in range(0,10):

    ax[i].imshow(train_x_noisy[i].reshape(28, 28))

plt.show()
input_layer = Input(shape=(28, 28, 1))



encoded_layer1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)

encoded_layer1 = MaxPool2D( (2, 2), padding='same')(encoded_layer1)

encoded_layer2 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded_layer1)

encoded_layer2 = MaxPool2D( (2, 2), padding='same')(encoded_layer2)

encoded_layer3 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded_layer2)

encoded_layer3 = MaxPool2D( (2, 2), padding='same')(encoded_layer3)



decoded_layer1 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded_layer3)

decoded_layer1 = UpSampling2D((2, 2))(decoded_layer1)

decoded_layer2 = Conv2D(32, (3, 3), activation='relu', padding='same')(decoded_layer1)

decoded_layer2 = UpSampling2D((2, 2))(decoded_layer2)

decoded_layer3 = Conv2D(64, (3, 3), activation='relu')(decoded_layer2)

decoded_layer3 = UpSampling2D((2, 2))(decoded_layer3)

output_layer   = Conv2D(1, (3, 3), padding='same')(decoded_layer3)



model = Model(input_layer, output_layer)

model.compile(optimizer='adam', loss='mse')
model_train = model.fit(train_x_noisy, train_x, epochs=80, batch_size=2048, validation_data=(val_x_noisy, val_x))
loss = model_train.history['loss']

val_loss = model_train.history['val_loss']

epochs = range(80)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training loss and Validation loss')

plt.legend()

plt.show()
model.save_weights('model_noisy_autoencoder.h5')
f, ax = plt.subplots(1,10)

for i in range(0,10):

    ax[i].imshow(val_x_noisy[i].reshape(28, 28))

plt.show()
preds = model.predict(val_x_noisy[:10])

f, ax = plt.subplots(1,10)

for i in range(0,10):

    ax[i].imshow(preds[i].reshape(28, 28))

plt.show()
train_labels_categorical = to_categorical(train_labels)
train_x,val_x,train_label,val_label = train_test_split(train_data,train_labels_categorical,test_size=0.2,random_state=13)



# train_data = train_data.reshape(-1, 28,28, 1)

train_x = train_x.reshape(-1, 28, 28, 1)

val_x = val_x.reshape(-1, 28, 28, 1)



train_x.shape,val_x.shape,train_label.shape,val_label.shape

num_classes = 10
def encoder(input_img):

    encoded_layer1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)

    encoded_layer1 = MaxPool2D( (2, 2), padding='same')(encoded_layer1)

    encoded_layer2 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded_layer1)

    encoded_layer2 = MaxPool2D( (2, 2), padding='same')(encoded_layer2)

    encoded_layer3 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded_layer2)

    encoded_layer3 = MaxPool2D( (2, 2), padding='same')(encoded_layer3)

    return encoded_layer3
def fully_connected(encoder):

    flat_layer = Flatten()(encoder)

    dense_layer = Dense(128, activation='relu')(flat_layer)

    out = Dense(num_classes, activation='softmax')(dense_layer)

    return out

model_classification = Model(input_layer,fully_connected(encoder(input_layer)))
for l1,l2 in zip(model_classification.layers[:7],model.layers[0:7]):

    l1.set_weights(l2.get_weights())

for layer in model_classification.layers[0:7]:

    layer.trainable = False

    

model_classification.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
classify_train = model_classification.fit(train_x, train_label, batch_size=64,epochs=80,verbose=1,validation_data=(val_x, val_label))
model_classification.save_weights('classification_encoder_model.h5')
for layer in model_classification.layers[0:7]:

    layer.trainable = True

    

model_classification.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
classify_train = model_classification.fit(train_x, train_label, batch_size=64,epochs=80,verbose=1,validation_data=(val_x, val_label))
model_classification.save_weights('classification_full_model.h5')
test_data = test_data.reshape(-1, 28, 28, 1)

test_label_predicted_classes = model_classification.predict(test_data)

test_label_predicted_classes = np.argmax(np.round(test_label_predicted_classes),axis=1)

correct_cases = np.where(test_label_predicted_classes==test_labels)[0]

print("Found %d correct labels", len(correct_cases))

for i, correct_cases in enumerate(correct_cases[:3]):

    plt.subplot(3,3,i+1)

    plt.imshow(test_data[correct_cases].reshape(28,28))

    plt.title("Predicted {}, Class {}".format(test_label_predicted_classes[correct_cases], test_labels[correct_cases]))

incorrect_cases = np.where(test_label_predicted_classes!=test_labels)[0]

print("Found %d incorrect labels" % len(incorrect_cases))

for i, incorrect_cases in enumerate(incorrect_cases[:3]):

    plt.subplot(3,3,i+1)

    plt.imshow(test_data[incorrect_cases].reshape(28,28))

    plt.title("Predicted {}, Class {}".format(test_label_predicted_classes[incorrect_cases], test_labels[incorrect_cases]))