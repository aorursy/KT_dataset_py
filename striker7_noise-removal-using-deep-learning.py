import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Loading Data
(x_train, labels_train), (x_test, labels_test) = tf.keras.datasets.fashion_mnist.load_data()

# Lets train rebscale the image 
x_train = x_train / 255.
x_test = x_test / 255.

# Any image have 3 dimensions width, height 
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

plt.imshow(x_train[0].reshape(28,28))
# adding noise 
sample_image = x_train[0]
noised_image = sample_image + 0.3*np.random.normal(0,1,sample_image.shape)
plt.imshow(noised_image.reshape(28,28))
x_noisy_train = x_train + 0.3 * np.random.normal(0,1,x_train.shape)
x_noisy_test = x_test + 0.3 * np.random.normal(0,1,x_test.shape)
plt.imshow(x_noisy_train[0].reshape(28,28))
encoder_input = tf.keras.layers.Input(shape=(28,28,1))
enc_layer = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(encoder_input)
norm = tf.keras.layers.BatchNormalization()(enc_layer)
enc_pool = tf.keras.layers.MaxPool2D()(norm)

enc_layer = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(enc_pool)
norm = tf.keras.layers.BatchNormalization()(enc_layer)
enc_pool = tf.keras.layers.MaxPool2D()(norm)

Encoder = tf.keras.Model(encoder_input, enc_pool)

Encoder.summary()
enc_pool.shape[1:]
decoder_input = tf.keras.layers.Input(shape=enc_pool.shape[1:])

dec_layer = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(decoder_input)
norm = tf.keras.layers.BatchNormalization()(dec_layer)
dec_pool = tf.keras.layers.UpSampling2D()(norm)

dec_layer = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(dec_pool)
norm = tf.keras.layers.BatchNormalization()(dec_layer)
dec_pool = tf.keras.layers.UpSampling2D()(norm)


decoded = tf.keras.layers.Conv2D(1, 3, activation='relu', padding='same')(dec_pool)
norm = tf.keras.layers.BatchNormalization()(decoded)

Decoder = tf.keras.Model(decoder_input, norm)

Decoder.summary()


dec_layer = tf.keras.layers.Conv2D(64,3,activation='relu', padding='same')(enc_pool)
norm = tf.keras.layers.BatchNormalization()(dec_layer)
dec_pool = tf.keras.layers.UpSampling2D()(norm)

dec_layer = tf.keras.layers.Conv2D(32,3,activation='relu', padding='same')(dec_pool)
norm = tf.keras.layers.BatchNormalization()(dec_layer)
dec_pool = tf.keras.layers.UpSampling2D()(norm)


decoded = tf.keras.layers.Conv2D(1, 3, activation='relu', padding='same')(dec_pool)

AutoEncoder = tf.keras.Model(encoder_input, decoded)

AutoEncoder.compile(optimizer='Adam', loss='mse')
with tf.device('device:GPU:0'):
    AutoEncoder.fit(x_noisy_train, x_train, batch_size=32, epochs=20)
clean_images = AutoEncoder.predict(x_noisy_test)
ax = plt.subplot(1,3,1)
plt.xlabel('original_image')
ax.imshow(x_test[2].reshape(28,28))

ax = plt.subplot(1,3,2)
plt.xlabel('noisy_image')
ax.imshow(x_noisy_train[2].reshape(28,28))

ax = plt.subplot(1,3,3)
plt.xlabel('image_after_noise_removal')
ax.imshow(clean_images[2].reshape(28,28))