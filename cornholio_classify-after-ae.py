import numpy as np

seed = sum(map(ord, 'irises_ae'))

np.random.seed(seed)
import pandas as pd

from keras.models import Model

from keras.layers import Input, Dense

from keras import regularizers
import matplotlib.pyplot as plt

%matplotlib inline
iris = pd.read_csv('../input/Iris.csv')

iris.head()
iris.tail()
v = iris.values

# fetch valuable columns

setosa = v[v[:, 5] == 'Iris-setosa'][:, 1:-1]

versicolor = v[v[:, 5] == 'Iris-versicolor'][:, 1:-1]

virginica = v[v[:, 5] == 'Iris-virginica'][:, 1:-1]

setosa.shape, versicolor.shape, virginica.shape
train_x = np.vstack((setosa, versicolor, virginica))

train_x.shape
# build autoencoder to 2d space

e_input = Input((4,))

encode = Dense(2, activation='sigmoid')(e_input)

decode = Dense(4)(encode)



ae = Model(inputs=[e_input], outputs=[decode])

ae.compile('adam', 'mse')
encoder = Model(e_input, encode) # separate model to encode data
end_input = Input((2,))

decoder_layer = ae.layers[-1]

decoder = Model(end_input, decoder_layer(end_input)) # separate model to decode data
m = ae.fit(train_x, train_x, epochs=1500, verbose=0) # train autoencoder
plt.plot(m.history['loss'])

plt.title('Loss')
e_iris = encoder.predict(train_x) # "translate" iris data to 2d space
plt.plot(e_iris[:50, 0], e_iris[:50, 1], 'co')

plt.plot(e_iris[50:100, 0], e_iris[50:100, 1], 'ro')

plt.plot(e_iris[100:150, 0], e_iris[100:150, 1], 'go')
c1 = np.zeros((50, 3))

c1[:, 0] = 1

c2 = np.zeros((50, 3))

c2[:, 1] = 1

c3 = np.zeros((50, 3))

c3[:, 2] = 1

print(c1[0], c2[0], c3[0])

e_iris_y = np.vstack((c1, c2, c3))

e_iris.shape
# classify 2d irises

c_input = Input((2,))

d = Dense(20, activation='sigmoid')(c_input)

d = Dense(18, activation='sigmoid')(d)

d = Dense(12, activation='sigmoid')(d)

d = Dense(6, activation='sigmoid')(d)

c_out = Dense(3, activation='softmax')(d)

c = Model(c_input, c_out)

c.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])
#m = c.fit(e_iris, e_iris_y, epochs=5000, validation_split=.2, verbose=0)

m = c.fit(e_iris, e_iris_y, epochs=5000, validation_split=.1, verbose=0)
print(min(m.history['loss']), min(m.history['val_loss']))

f, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].plot(m.history['loss'])

ax[0].set_title('Train loss')

ax[1].plot(m.history['val_loss'])

ax[1].set_title('Validation loss')
print(max(m.history['acc']), max(m.history['val_acc']))

f, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].plot(m.history['acc'])

ax[0].set_title('Train accuracy')

ax[1].plot(m.history['val_acc'])

ax[1].set_title('Validation accuracy')