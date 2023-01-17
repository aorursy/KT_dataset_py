import numpy as np
np.random.seed(0)
from keras import backend as K
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras.utils import np_utils
import matplotlib.pyplot as plt
%matplotlib inline
(x_train_orig, _), (x_test_orig, _) = mnist.load_data()
x_train_orig.shape, x_test_orig.shape
plt.imshow(x_train_orig[0], cmap='Greys_r')
n = 15  # how many digits we will display
plt.figure(figsize=(n*2, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train_orig[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
img = x_train_orig[0]
img.min(), img.max()
x_train_orig = x_train_orig.astype('float32') / 255.
x_test_orig = x_test_orig.astype('float32') / 255.
n = 15  # how many digits we will display
plt.figure(figsize=(n*2, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train_orig[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
img = x_train_orig[0]
img.min(), img.max()
noise_factor = 0.5
x_train_noisy = np.clip(x_train_orig + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_orig.shape), 0, 1)
x_test_noisy = np.clip(x_test_orig + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_orig.shape), 0, 1)
n = 15  # how many digits we will display
plt.figure(figsize=(n*2, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train_orig[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noisy
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_train_noisy[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
x_train_orig.shape, x_train_noisy.shape
x_train = x_train_orig.reshape((len(x_train_orig), np.prod(x_train_orig.shape[1:])))
x_test = x_test_orig.reshape((len(x_test_orig), np.prod(x_test_orig.shape[1:])))
x_train.shape
x_train_conv = x_train_orig.reshape(-1,28,28,1)
x_test_conv = x_test_orig.reshape(-1,28,28,1)
x_train_noisy_conv = x_train_noisy.reshape(-1,28,28,1)
x_test_noisy_conv = x_test_noisy.reshape(-1,28,28,1)
x_train_conv.shape
encoding_dim = 32
image_dim = x_train.shape[1]
image_dim
input_img = Input(shape=(image_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(image_dim, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
# carregando o modelo, se necessário
#autoencoder.load_weights('checkpoint_modelo_1.hdf5')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()
checkpointer_1 = ModelCheckpoint(filepath='checkpoint_modelo_1.hdf5', verbose=1, save_best_only=True)
history_model_1 = autoencoder.fit(x_train, x_train,
                                  epochs=70,
                                  batch_size=256,
                                  shuffle=True,
                                  validation_data=(x_test, x_test),
                                  callbacks=[checkpointer_1])
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
n = 15  # how many digits we will display
plt.figure(figsize=(2*n, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# Grafico do treinamento
plt.plot(history_model_1.history['loss'], 'b')
plt.plot(history_model_1.history['val_loss'], 'r')
plt.show()
input_img = Input(shape=(image_dim,))

encoded_reg = Dense(32, activation='relu', activity_regularizer=regularizers.l1(5*10e-8))(input_img)
decoded_reg = Dense(784, activation='sigmoid')(encoded_reg)

autoencoder_reg = Model(input_img, decoded_reg)
# carregando o modelo, se necessário
#autoencoder.load_weights('checkpoint_modelo_2.hdf5')
autoencoder_reg.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder_reg.summary()
checkpointer_2 = ModelCheckpoint(filepath='checkpoint_modelo_2.hdf5', verbose=1, save_best_only=True)
history_model_2 = autoencoder_reg.fit(x_train, x_train,
                                    epochs=300,
                                    batch_size=256,
                                    shuffle=True,
                                    validation_data=(x_test, x_test),
                                    callbacks=[checkpointer_2])
# Grafico do treinamento
plt.plot(history_model_2.history['loss'], 'b')
plt.plot(history_model_2.history['val_loss'], 'r')
plt.show()
encoder_reg = Model(input_img, encoded_reg)
#encoded_input = Input(shape=(encoding_dim,))
decoder_layer_reg = autoencoder_reg.layers[-1]
decoder_reg = Model(encoded_input, decoder_layer_reg(encoded_input))
encoded_imgs_reg = encoder_reg.predict(x_test)
decoded_imgs_reg = decoder_reg.predict(encoded_imgs_reg)
n = 15  # how many digits we will display
plt.figure(figsize=(2*n, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_reg[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
encoded_imgs.mean()
encoded_imgs_reg.mean()
#input_img = Input(shape=(image_dim,))

encoded_deep = Dense(128, activation='relu')(input_img)
encoded_deep = Dense(64, activation='relu')(encoded_deep)
encoded_deep = Dense(32, activation='relu')(encoded_deep)

decoded_deep = Dense(64, activation='relu')(encoded_deep)
decoded_deep = Dense(128, activation='relu')(decoded_deep)
decoded_deep = Dense(784, activation='sigmoid')(decoded_deep)

autoencoder_deep = Model(input_img, decoded_deep)
# carregando o modelo, se necessário
#autoencoder.load_weights('checkpoint_modelo_3.hdf5')
autoencoder_deep.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder_deep.summary()
checkpointer_3 = ModelCheckpoint(filepath='checkpoint_modelo_3.hdf5', verbose=1, save_best_only=True)
history_model_3 = autoencoder_deep.fit(x_train, x_train,
                                        epochs=200,
                                        batch_size=256,
                                        shuffle=True,
                                        validation_data=(x_test, x_test),
                                        callbacks=[checkpointer_3])
# Grafico do treinamento
plt.plot(history_model_3.history['loss'], 'b')
plt.plot(history_model_3.history['val_loss'], 'r')
plt.show()
encoder_deep = Model(input_img, encoded_deep)
#encoded_input = Input(shape=(encoding_dim,))

decoder_layer_deep = autoencoder_deep.layers[-3]
decoder_deep = Model(encoded_input, decoder_layer_deep(encoded_input))
decoder_deep_new = Sequential()
decoder_deep_new.add(decoder_deep)
decoder_deep_new.add(autoencoder_deep.layers[-2])
decoder_deep_new.add(autoencoder_deep.layers[-1])
decoded_imgs_deep = autoencoder_deep.predict(x_test)
n = 15  # how many digits we will display
plt.figure(figsize=(2*n, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_deep[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
K.image_data_format()
input_img_conv = Input(shape=(28, 28, 1))

encoded_conv = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img_conv)
encoded_conv = MaxPooling2D((2, 2), padding='same')(encoded_conv)
encoded_conv = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_conv)
encoded_conv = MaxPooling2D((2, 2), padding='same')(encoded_conv)
encoded_conv = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_conv)
encoded_conv = MaxPooling2D((2, 2), padding='same')(encoded_conv)

# (4, 4, 8) => 128

decoded_conv = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_conv)
decoded_conv = UpSampling2D((2, 2))(decoded_conv)
decoded_conv = Conv2D(8, (3, 3), activation='relu', padding='same')(decoded_conv)
decoded_conv = UpSampling2D((2, 2))(decoded_conv)
decoded_conv = Conv2D(16, (3, 3), activation='relu')(decoded_conv)
decoded_conv = UpSampling2D((2, 2))(decoded_conv)
decoded_conv = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded_conv)
autoencoder_conv = Model(input_img_conv, decoded_conv)
# carregando o modelo, se necessário
#autoencoder.load_weights('checkpoint_modelo_4.hdf5')
autoencoder_conv.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder_conv.summary()
checkpointer_4 = ModelCheckpoint(filepath='checkpoint_modelo_4.hdf5', verbose=1, save_best_only=True)
history_model_4 = autoencoder_conv.fit(x_train_conv, x_train_conv,
                                        epochs=100,
                                        batch_size=256,
                                        shuffle=True,
                                        validation_data=(x_test_conv, x_test_conv),
                                        callbacks=[checkpointer_4])
# Grafico do treinamento
plt.plot(history_model_4.history['loss'], 'b')
plt.plot(history_model_4.history['val_loss'], 'r')
plt.show()
decoded_imgs_conv = autoencoder_conv.predict(x_test_conv)
n = 15  # how many digits we will display
plt.figure(figsize=(2*n, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_conv[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
input_img_conv = Input(shape=(28, 28, 1))

encoded_conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img_conv)
encoded_conv2 = MaxPooling2D((2, 2), padding='same')(encoded_conv2)
encoded_conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_conv2)
encoded_conv2 = MaxPooling2D((2, 2), padding='same')(encoded_conv2)
encoded_conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_conv2)
encoded_conv2 = MaxPooling2D((2, 2), padding='same')(encoded_conv2)

# (4, 4, 8) => 128

decoded_conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_conv2)
decoded_conv2 = UpSampling2D((2, 2))(decoded_conv2)
decoded_conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(decoded_conv2)
decoded_conv2 = UpSampling2D((2, 2))(decoded_conv2)
decoded_conv2 = Conv2D(16, (3, 3), activation='relu')(decoded_conv2)
decoded_conv2 = UpSampling2D((2, 2))(decoded_conv2)
decoded_conv2 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded_conv2)


autoencoder_conv_denoise = Model(input_img_conv, decoded_conv2)
# carregando o modelo, se necessário
#autoencoder.load_weights('checkpoint_modelo_5.hdf5')
autoencoder_conv_denoise.compile(optimizer='adadelta', loss='binary_crossentropy')
checkpointer_5 = ModelCheckpoint(filepath='checkpoint_modelo_5.hdf5', verbose=1, save_best_only=True)
history_model_5 = autoencoder_conv_denoise.fit(x_train_noisy_conv, x_train_conv,
                                          epochs=100,
                                          batch_size=256,
                                          shuffle=True,
                                          validation_data=(x_test_noisy_conv, x_test_conv),
                                          callbacks=[checkpointer_4])
# Grafico do treinamento
plt.plot(history_model_5.history['loss'], 'b')
plt.plot(history_model_5.history['val_loss'], 'r')
plt.show()
decoded_imgs_conv_denoise = autoencoder_conv_denoise.predict(x_test_noisy_conv)
n = 15  # how many digits we will display
plt.figure(figsize=(2*n, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_conv_denoise[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

