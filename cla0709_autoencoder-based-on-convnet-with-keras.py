import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from skimage import data,io,filters
from sklearn import preprocessing

from keras import Sequential
from keras.layers.core import Dense,Flatten,Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD
from keras.layers import Input,Reshape
from keras.models import Model

import cv2
df = pd.read_csv("../input/train.csv")
y = df['label'].values
X = (1 - df.iloc[:,1:].values/255).reshape(-1,28,28)
X = np.expand_dims(X, axis=3)
input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
print(input_img.get_shape)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
print(x.get_shape)
x = MaxPooling2D((2, 2), padding='same')(x)
print(x.get_shape)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
print(x.get_shape)
x = MaxPooling2D((2, 2), padding='same')(x)
print(x.get_shape)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
print(x.get_shape)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
print(x.get_shape)
x = UpSampling2D((2, 2))(x)
print(x.get_shape)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
print(x.get_shape)
x = UpSampling2D((2, 2))(x)
print(x.get_shape)
x = Conv2D(16, (3, 3), activation='relu')(x)
print(x.get_shape)
x = UpSampling2D((2, 2))(x)
print(x.get_shape)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

print(decoded.get_shape)
autoencoder = Model(input_img,decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
from keras.callbacks import TensorBoard

autoencoder.fit(X, X,
                epochs=10,
                batch_size=128,
                shuffle=True,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
import matplotlib.pyplot as plt

decoded_imgs = autoencoder.predict(X[0:10])

n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(X[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
encoder = Sequential()
for layer in autoencoder.layers[0:7]:
    encoder.add(layer)

decoder = Sequential()
decoder.add(Reshape(input_shape=(4,4,8), target_shape=(4,4,8)))
for layer in autoencoder.layers[7:]:
    print(layer.output_shape)
    decoder.add(layer)
x_0 = X[10]
print(y[10])
plt.figure()
plt.subplot(131)
plt.imshow(x_0.reshape(28,28))
x_0 = np.expand_dims(x_0, axis=0)
x_enc = encoder.predict(x_0)
new_x = decoder.predict(x_enc)
plt.subplot(132)
plt.imshow(x_enc.reshape(16, 8))
plt.subplot(133)
plt.imshow(new_x.reshape(28, 28))
from sklearn.decomposition import PCA

numbers_avg = np.zeros((10,128))
numbers_occ = np.zeros(10)

X_enc = encoder.predict(X).reshape(-1,128)

PCA_ = PCA(n_components = 128).fit(X_enc)
X_enc_pca = PCA_.transform(X_enc)
for i, number in enumerate(y):
    numbers_occ[number] += 1
    numbers_avg[number] += X_enc_pca[i]

for i in range(0,10):
    numbers_avg[i] /= numbers_occ[i]
n = 11
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(1, n, i)
    # display reconstruction
    plt.imshow(decoder.predict(PCA_.inverse_transform(numbers_avg[i-1]).reshape(1,4,4,8)).reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

from sklearn.manifold import TSNE

tsne = TSNE()
x_tsne = tsne.fit_transform(X_enc_pca[0:2000])
classes = set(y)
colors = [plt.cm.tab10(float(i)/max(classes)) for i in classes]
for i, c in enumerate(classes):
    xi = [x_tsne[j,0] for j  in range(len(x_tsne)) if y[j] == c]
    yi = [x_tsne[j,1] for j  in range(len(x_tsne)) if y[j] == c]
    plt.scatter(xi, yi, c=colors[i], label=str(c))
plt.legend()
plt.show()
mutant = np.zeros(128)
n = 11
plt.figure(figsize=(20, 20))
for i in range(1,n):
    mutant[0] = 10*(i-1)
    for j in range(1,n):
        # display original
        ax = plt.subplot(n, n, (i-1)*n+j)
        # display reconstruction
        mutant[1] = 10*(j-1)
        plt.imshow(decoder.predict(PCA_.inverse_transform(mutant).reshape(1,4,4,8)).reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()

