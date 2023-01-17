# Crearea unei funcții pentru recunoașterea numelor fișierelor de date din setul MNIST

import os

import struct

import numpy as np

 

def load_mnist(path, kind='train'):

    """Încărcarea setului de date MNIST din cale"""

    labels_path = os.path.join(

        path, f'{kind}-labels-idx1-ubyte'

    )

    images_path = os.path.join(

        path, f'{kind}-images-idx3-ubyte'

    )

        

    with open(labels_path, 'rb') as lbpath:

        magic, n = struct.unpack('>II', lbpath.read(8))

        labels = np.fromfile(lbpath, dtype=np.uint8)

 

    with open(images_path, 'rb') as imgpath:

        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))

        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

        images = ((images / 255.) - .5) * 2

 

    return images, labels
# Încărcarea datelor

X_train, y_train = load_mnist('../input/', kind='train')

print(f'Rows: {X_train.shape[0]},  Columns: {X_train.shape[1]}')



X_test, y_test = load_mnist('../input/', kind='t10k')

print(f'Rows: {X_test.shape[0]},  Columns: {X_test.shape[1]}')



# Centrarea după medie și normalizarea datelor

mean_vals = np.mean(X_train, axis=0)

std_val = np.std(X_train)



X_train_centered = (X_train - mean_vals)/std_val

X_test_centered = (X_test - mean_vals)/std_val



del X_train, X_test



print(X_train_centered.shape, y_train.shape)

print(X_test_centered.shape, y_test.shape)
# Vizualizarea cifrelor de la 0 la 9 din setul de date MNIST

import matplotlib.pyplot as plt



fig, ax = plt.subplots(nrows=2, ncols=5,

                       sharex=True, sharey=True)

ax = ax.flatten()

for i in range(10):

    img = X_train_centered[y_train == i][0].reshape(28, 28)

    ax[i].imshow(img, cmap='Greys')



ax[0].set_yticks([])

ax[0].set_xticks([])

plt.tight_layout()

plt.show()
import tensorflow as tf

import tensorflow.contrib.keras as keras



np.random.seed(123)

tf.set_random_seed(123)



y_train_onehot = keras.utils.to_categorical(y_train)



print('First 3 labels: ', y_train[:3])

print('\nFirst 3 labels (one-hot):\n', y_train_onehot[:3])
# Inițializarea modelului în varabila model

model = keras.models.Sequential()



# Adăugarea stratului de intrare

model.add(keras.layers.Dense(

    units=50,

    input_dim=X_train_centered.shape[1],

    kernel_initializer='glorot_uniform',

    bias_initializer='zeros',

    activation='tanh') 

)



# Adăugarea stratului ascuns

model.add(

    keras.layers.Dense(

        units=50,

        input_dim=50,

        kernel_initializer='glorot_uniform',

        bias_initializer='zeros',

        activation='tanh')

    )



# Adăugarea stratului de ieșire

model.add(

    keras.layers.Dense(

        units=y_train_onehot.shape[1],

        input_dim=50,

        kernel_initializer='glorot_uniform',

        bias_initializer='zeros',

        activation='softmax')

    )



# Definirea optimizatorului SGD

sgd_optimizer = keras.optimizers.SGD(

    lr=0.001, decay=1e-7, momentum=0.9

)



# Compilarea modelului

model.compile(

    optimizer=sgd_optimizer,

    loss='categorical_crossentropy'

)
# Antrenarea modelului

history = model.fit(

    X_train_centered, y_train_onehot,

    batch_size=64, epochs=50,

    verbose=1, validation_split=0.1

)
y_train_pred = model.predict_classes(X_train_centered, verbose=0)

print('Primele zece predicții: ', y_train_pred[:10])
# Calcularea acurateței modelului pe setul de antrenare

y_train_pred = model.predict_classes(X_train_centered, verbose=0)

correct_preds = np.sum(y_train == y_train_pred, axis=0)

train_acc = correct_preds / y_train.shape[0]



print(f'Acuratețea modelului pe setul de antrenare: {(train_acc * 100):.2f}')



# Calcularea acurateței modelului pe setul de test

y_test_pred = model.predict_classes(X_test_centered, verbose=0)

correct_preds = np.sum(y_test == y_test_pred, axis=0)

test_acc = correct_preds / y_test.shape[0]



print(f'Acuratețea modelului pe setul de test: {(test_acc * 100):.2f}')