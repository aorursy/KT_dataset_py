import pandas as pd

train_data = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')
test_data = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_test.csv')

# X
train_labels = train_data.label
test_labels = test_data.label
# y
train_images = train_data.iloc[:, 1:].to_numpy()
test_images = test_data.iloc[:, 1:].to_numpy()

print('Train size : ' , train_labels.shape)
print('Test  size : ', test_labels.shape)
# For X 

import numpy as np

# 1. Reshape 
train_images = train_images.reshape((60000, 28, 28))
test_images = test_images.reshape((10000, 28, 28))


# 2. Sacle
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

# 3. Bigger size. (becuase image is 2D, we have to repeat on axises 1 and 2 )
train_images = np.repeat(train_images, 2, axis=1) 
train_images = np.repeat(train_images, 2, axis=2)

test_images = np.repeat(test_images, 2, axis=1)
test_images = np.repeat(test_images, 2, axis=2)

# 4. GrayScale to  RGB
train_images = np.stack((train_images,) * 3, axis=-1)
test_images = np.stack((test_images,) * 3, axis=-1)


# 5. Valid 
valid_images = train_images[50000:]
train_images = train_images[:50000]
# For y

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
valid_labels = train_labels[50000:]
train_labels = train_labels[:50000]
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Add
from keras.models import Model

# VGG16 + Skip Connection
_input = Input((56,56,3)) 

conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(_input)
conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)
pool1  = MaxPooling2D((2, 2))(conv2)

shortcut = pool1 # Skip connection
shortcut = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(shortcut)
shortcut  = MaxPooling2D((16, 16))(shortcut)


conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)
pool2  = MaxPooling2D((2, 2))(conv4)

conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)
conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)
pool3  = MaxPooling2D((2, 2))(conv7)

conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)
conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)
pool4  = MaxPooling2D((2, 2))(conv10)

conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)
conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)
pool5  = MaxPooling2D((2, 2))(conv13)

add = Add()([shortcut, pool5]) # Skip connection joined

flat   = Flatten()(add)
dense1 = Dense(4096, activation="relu")(flat)
dense2 = Dense(4096, activation="relu")(dense1)
output = Dense(10, activation="softmax")(dense2)

vgg16_skip_model  = Model(inputs=_input, outputs=output)

# VGG16 
_input = Input((56,56,3)) 

conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(_input)
conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)
pool1  = MaxPooling2D((2, 2))(conv2)


conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)
pool2  = MaxPooling2D((2, 2))(conv4)

conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)
conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)
pool3  = MaxPooling2D((2, 2))(conv7)

conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)
conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)
pool4  = MaxPooling2D((2, 2))(conv10)

conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)
conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)
pool5  = MaxPooling2D((2, 2))(conv13)

flat   = Flatten()(pool5)
dense1 = Dense(4096, activation="relu")(flat)
dense2 = Dense(4096, activation="relu")(dense1)
output = Dense(10, activation="softmax")(dense2)

vgg16_model  = Model(inputs=_input, outputs=output)


# CNN 
_input = Input((56,56,3)) 

conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(_input)
conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)
pool1  = MaxPooling2D((2, 2))(conv2)

flat   = Flatten()(pool1)
dense1 = Dense(4096, activation="relu")(flat)
dense2 = Dense(4096, activation="relu")(dense1)
output = Dense(10, activation="softmax")(dense2)

cnn_model  = Model(inputs=_input, outputs=output)

# Dense
_input = Input((56,56,3)) 
flat   = Flatten()(_input)
dense1 = Dense(4096, activation="relu")(flat)
dense2 = Dense(4096, activation="relu")(dense1)
output = Dense(10, activation="softmax")(dense2)

dense_model  = Model(inputs=_input, outputs=output)
#vgg16_model.summary()
# Training 
from keras import optimizers

loss = 'categorical_crossentropy'
metrics = ['accuracy']
optimizer = optimizers.RMSprop(lr=1e-5)
batch_size = 128
epochs = 5

#VGG16_skip
vgg16_skip_model.compile(optimizer= optimizer,
                loss=loss,
                metrics = metrics)


history1 = vgg16_skip_model.fit(x=train_images, y=train_labels,
                        validation_data=(valid_images, valid_labels),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)

#VGG16
vgg16_model.compile(optimizer= optimizer,
                loss=loss,
                metrics = metrics)


history2 = vgg16_model.fit(x=train_images, y=train_labels,
                        validation_data=(valid_images, valid_labels),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)

#CNN
cnn_model.compile(optimizer= optimizer,
                loss=loss,
                metrics = metrics)


history3 = cnn_model.fit(x=train_images, y=train_labels,
                        validation_data=(valid_images, valid_labels),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)

#Dense
dense_model.compile(optimizer= optimizer,
                loss=loss,
                metrics = metrics)


history4 = dense_model.fit(x=train_images, y=train_labels,
                        validation_data=(valid_images, valid_labels),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)
import matplotlib.pyplot as plt

histories = [history1, history2, history3, history4]
colors = ['red','blue','green', 'orange']
legends = ['VGG16+skip', 'VGG16','CNN', 'Dense']
f =  plt.figure(figsize=(10,10))

for i, history in enumerate(histories):

    try:
          acc = history.history['accuracy']
          val_acc = history.history['val_accuracy']
    except:
          acc = history.history['acc']
          val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)


    plt.plot(epochs, acc,'--', color=colors[i], label=legends[i]+'Train')
    plt.plot(epochs, val_acc, color=colors[i], label=legends[i]+'Validation')
    plt.title('Validation accuracy')
    plt.legend()

test_loss, test_acc = vgg16_skip_model.evaluate(test_images, test_labels, verbose=2)
print("vgg16_skip", "test_acc", test_acc, "test_loss", test_loss)

test_loss, test_acc = vgg16_model.evaluate(test_images, test_labels, verbose=2)
print("vgg16     ", "test_acc", test_acc, "test_loss", test_loss)

test_loss, test_acc = cnn_model.evaluate(test_images, test_labels, verbose=2)
print("cnn       ", "test_acc", test_acc, "test_loss", test_loss)

test_loss, test_acc = dense_model.evaluate(test_images, test_labels, verbose=2)
print("dense     ", "test_acc", test_acc, "test_loss", test_loss)