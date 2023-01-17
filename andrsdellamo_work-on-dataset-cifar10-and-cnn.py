from tensorflow import keras as ks
from matplotlib import pyplot as plt
import numpy as np
import time
import datetime

from keras import models
from tensorflow import keras as ks
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2 
num_classes=10
# Creacion del modelo
model = ks.Sequential()

model.add(ks.layers.Conv2D(32, (3, 3),strides=1, input_shape=(32, 32,3), padding='same', activation='relu'))
model.add(ks.layers.Conv2D(32, (3, 3),strides=1, activation='relu', padding='same'))
model.add(ks.layers.Conv2D(32, (3, 3),strides=1, activation='relu', padding='valid'))
model.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(ks.layers.Dropout(0.2))

model.add(ks.layers.Flatten())
model.add(ks.layers.Dense(512, activation='relu'))
model.add(ks.layers.Dropout(0.5))
model.add(ks.layers.Dense(512, activation='relu'))
model.add(ks.layers.Dropout(0.5))
model.add(ks.layers.Dense(num_classes, activation='softmax'))
model.summary()
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callback_val_loss = EarlyStopping(monitor="val_loss", patience=10)
callback_val_accuracy = EarlyStopping(monitor="val_accuracy", patience=10)
cifar10 = ks.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
cifar10_labels = [
'airplane', # id 0
'automobile',
'bird',
'cat',
'deer',
'dog',
'frog',
'horse',
'ship',
'truck',
]

print('Number of labels: %s' % len(cifar10_labels))
# Pintemos una muestra de las las imagenes del dataset MNIST

print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

for i in range(9):

    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    plt.title(cifar10_labels[y_train[i,0]])

plt.subplots_adjust(hspace = 1)
plt.show()
x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]
x_train_cnn = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test_cnn = x_test.reshape((x_test.shape[0], 32, 32, 3))
x_val_cnn = x_val.reshape((x_val.shape[0],32,32,3))

# Validamos el resultado
print('Train: X=%s, y=%s' % (x_train_cnn.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test_cnn.shape, y_test.shape))
print('Validation: X=%s, y=%s' % (x_val_cnn.shape, y_val.shape))
t = time.perf_counter()
history = model.fit(x_train_cnn, y_train, epochs=300, use_multiprocessing=False, 
                    batch_size= 512, validation_data=(x_val, y_val),
                    callbacks=[callback_val_loss, callback_val_accuracy] )
_, acc = model.evaluate(x_test, y_test, verbose=0)
print('> %.3f' % (acc * 100.0))
plt.title('Cross Entropy Loss')
plt.plot(history.history['loss'], color='blue', label='train')
plt.plot(history.history['val_loss'], color='orange', label='test')
plt.show()

plt.title('Classification Accuracy')
plt.plot(history.history['accuracy'], color='blue', label='train')
plt.plot(history.history['val_accuracy'], color='orange', label='test')
plt.show()
layer_outputs = [layer.output for layer in model.layers[:4]] 
layer_outputs

activation_model = models.Model(inputs=model.input, outputs=layer_outputs) 
imagen= x_train_cnn[0,:,:,:]
plt.imshow(imagen)
img_tensor = np.expand_dims(imagen, axis=0)
plt.imshow(img_tensor[0])
plt.show()
print(img_tensor.shape)
activations = activation_model.predict( img_tensor)
fig = plt.figure( figsize=(16,16))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for i in range(0, 32):
    ax = fig.add_subplot(8, 4, i+1)
    ax.imshow(activations[0][0, :, :, i], cmap=plt.get_cmap('viridis'))
fig = plt.figure( figsize=(16,16))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for i in range(0, 32):
    ax = fig.add_subplot(8, 4, i+1)
    ax.imshow(activations[1][0, :, :, i], cmap=plt.get_cmap('viridis'))
fig = plt.figure( figsize=(16,16))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for i in range(0, 32):
    ax = fig.add_subplot(8, 4, i+1)
    ax.imshow(activations[2][0, :, :, i], cmap=plt.get_cmap('viridis'))
fig = plt.figure( figsize=(16,16))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for i in range(0, 32):
    ax = fig.add_subplot(8, 4, i+1)
    ax.imshow(activations[3][0, :, :, i], cmap=plt.get_cmap('viridis'))
