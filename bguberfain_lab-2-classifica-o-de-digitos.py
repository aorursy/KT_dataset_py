import numpy as np
np.random.seed(5)

#from keras.datasets import mnist
from keras.models import Sequential
from keras import layers
from keras.utils import np_utils
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

(X_train, y_train), (X_test, y_test) = load_data('../input/mnist-numpy/mnist.npz')
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

fig, ax = plt.subplots(ncols=10, nrows=1, figsize=(10, 5))
amostra = np.random.choice(60000, 10) #escolhe 10 imagens dentre as 60000

for i in range(len(amostra)):
    imagem = np.array(X_train[amostra[i]])
    ax[i].imshow(imagem, cmap = cm.Greys_r)
    ax[i].get_xaxis().set_ticks([])
    ax[i].get_yaxis().set_ticks([])
    ax[i].set_title(y_train[amostra[i]]) # Coloca o label como título da figura.
plt.show()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
n_classes = 10 #são 10 classes: números de 0 a 9
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
X_train_flat = X_train.reshape(60000, 784)
X_test_flat = X_test.reshape(10000, 784)
#Parâmetros
nb_epoch = 15
batch_size = 128
model = Sequential()
model.add(layers.Dense(512, input_shape=(784,)))
model.add(layers.Activation('relu'))
model.add(layers.Dense(10))
model.add(layers.Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_flat, Y_train,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=1, validation_data=(X_test_flat, Y_test))

score = model.evaluate(X_test_flat, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model2 = Sequential()
model2.add(layers.Dense(512, input_shape=(784,)))
model2.add(layers.Activation('relu'))
## Nova camada
model2.add(layers.Dense(512))
model2.add(layers.Activation('relu'))

model2.add(layers.Dense(10))
model2.add(layers.Activation('softmax'))

model2.summary()

model2.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model2.fit(X_train_flat, Y_train,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=1, validation_data=(X_test_flat, Y_test))

score = model2.evaluate(X_test_flat, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model3 = Sequential()
model3.add(layers.Dense(512, input_shape=(784,)))
model3.add(layers.Activation('relu'))
model3.add(layers.Dropout(0.3)) # percentual de neurônios que serão zerados durante o aprendizado
model3.add(layers.Dense(512))
model3.add(layers.Activation('relu'))
model3.add(layers.Dropout(0.3)) # percentual de neurônios que serão zerados durante o aprendizado
model3.add(layers.Dense(10))
model3.add(layers.Activation('softmax'))

model3.summary()

model3.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model3.fit(X_train_flat, Y_train,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=1, validation_data=(X_test_flat, Y_test))

score = model3.evaluate(X_test_flat, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
img_rows = 28
img_cols = 28

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
## Parâmetros
batch_size = 2048
n_filters = 32 #número de filtros
n_pool = 2 #Tamanho da camada de pooling
n_conv = 3 #Tamanho da kernel do filtro 
model4 = Sequential()
model4.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model4.add(layers.Dropout(0.5))
model4.add(layers.MaxPooling2D((2, 2)))
model4.add(layers.Conv2D(64, (3, 3), activation='relu'))
model4.add(layers.Dropout(0.5))
model4.add(layers.MaxPooling2D((2, 2)))
model4.add(layers.Conv2D(64, (3, 3), activation='relu'))
model4.add(layers.Dropout(0.5))
model4.add(layers.Flatten())
model4.add(layers.Dense(64, activation='relu'))
model4.add(layers.Dense(10, activation='softmax'))
model4.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model4.summary()

model4.fit(X_train, Y_train, batch_size=batch_size, epochs=30, 
          verbose=1, validation_data=(X_test, Y_test))
score = model4.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])