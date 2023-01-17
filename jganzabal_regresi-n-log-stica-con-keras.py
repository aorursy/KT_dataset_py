from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint 

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.normalization import BatchNormalization

from keras import optimizers

from keras import initializers

import numpy as np

from matplotlib import pyplot as plt
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
folder = '/kaggle/input/fashion-mnist-itba-lab-2020/'

x = np.load(folder+'train_images.npy')

y = np.loadtxt(folder+'train_labels.csv', delimiter=',', skiprows=1)

x_test = np.load(folder+'test_images.npy')
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.1)
y_train_categorical = to_categorical(y_train)

y_val_categorical = to_categorical(y_valid)
alto = 4

ancho = 8

f, axs = plt.subplots(alto, ancho, figsize=(30,4*alto))

axs = axs.reshape(-1)

for i in range(alto*ancho):

    axs[i].imshow(x_train[i], cmap='gray')

    axs[i].set_title(f'{classes[int(y_train[i])]} - class: {int(y_train[i])}\n{y_train_categorical[i]}')

    axs[i].axis('off')
output_size = 10

model_single_layer = Sequential()

model_single_layer.add(Flatten(input_shape=x_train.shape[1:]))

model_single_layer.add(Dense(output_size, name='Salida'))

model_single_layer.add(Activation('softmax'))

model_single_layer.summary()
lr = 0.000001 

SGD = optimizers.sgd(lr=lr)

model_single_layer.compile(loss = 'categorical_crossentropy', optimizer=SGD, metrics=['accuracy'])
batch_size = 512

model_single_layer.fit(x_train, 

                       y_train_categorical,

                       epochs=20, batch_size=batch_size, 

                       verbose=1, 

                       validation_data = (x_valid, y_val_categorical)

                      )
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))

ax1.plot(model_single_layer.history.history['loss'], label='train')

ax1.plot(model_single_layer.history.history['val_loss'], label='val')

ax1.set_title('Loss - Cross Entropy')

ax1.legend()

ax2.plot(model_single_layer.history.history['accuracy'], label='train')

ax2.plot(model_single_layer.history.history['val_accuracy'], label='val')

ax2.set_title('Metric - Accuracy')

ax2.legend()

plt.show()
loss, acc = model_single_layer.evaluate(x_valid, y_val_categorical, verbose=0)

print(acc, loss)
# Calculo probabilidades de cada clase para cada observación

test_prediction = model_single_layer.predict(x_test)

print(test_prediction.shape)
# Calculo clase

test_labels = np.argmax(test_prediction, axis = 1)

print(test_labels)
import pandas

df = pandas.DataFrame(data={"Category": test_labels}).astype(int)

df.to_csv("./submission.csv", sep=',',index=True,  index_label='Id')