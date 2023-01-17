# imports
import os
import datetime
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from matplotlib import pyplot as plt
@dataclass
class Data:
    
    train: tf.data.Dataset
    test:  tf.data.Dataset
    label_mapping = {
        # comes from https://www.tensorflow.org/tutorials/keras/classification
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }
    
    @classmethod
    def get_mnist(cls):
        # mnist or fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        return cls(train_ds, test_ds)
    
    @classmethod
    def map_labels(cls, labels: list):
        return [cls.label_mapping[l] if l in cls.label_mapping else 'unknown' for l in labels]

    
mnist_data = Data.get_mnist()
image_samples = mnist_data.train.as_numpy_iterator()
plt.figure(figsize=(16, 10))
for i in range(100):
	img, _ = next(image_samples)
	plt.subplot(10, 10, 1 + i)
	plt.axis('off')
	plt.imshow(img, cmap='gray_r')
plt.show();
# utils
def preprocess(images, labels):
    x = tf.cast(images, dtype=tf.float32)
    x = (x-127.5)/127.5
    return tf.expand_dims(x, axis=-1), labels

def postprocess(images):
    return tf.squeeze((images*127.5)+127.5)
# DISCRIMINATOR
def define_discriminator():
    """A fairly standard CNN classifier"""
    optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding="same", input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding="same"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

# GENERATOR -- this is ultimately what we are interested in
def define_generator():
    """Generate an image given random input"""
    # NOTE: we do not compile this model within the definition
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*128, input_dim=100))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Reshape((7,7,128)))
    model.add(tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Conv2D(1, (7,7), activation="tanh", padding="same"))
    return model


def define_gan(generator, discriminator):
    """GAN model used to train the generator only"""
    optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    discriminator.trainable = False # discriminator is trained externally
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)
    # generator feeds into discriminator and we update generator based on how well it performed
    return model


BATCH_SIZE=100
EPOCHS=50

discriminator = define_discriminator()
generator = define_generator()
gan = define_gan(generator, discriminator)

# training
for e in range(EPOCHS):
    for images, labels in mnist_data.train.map(preprocess).batch(BATCH_SIZE):
        noise = tf.random.normal([BATCH_SIZE, 100])
        fake_images = generator.predict(noise)
        discriminator_loss_fake, _ = discriminator.train_on_batch(fake_images, tf.zeros(BATCH_SIZE))
        discriminator_loss_true, _ = discriminator.train_on_batch(images, tf.ones(BATCH_SIZE))
        gan_loss = gan.train_on_batch(noise, tf.ones(BATCH_SIZE))
    if (e % 10) == 0:
        print("Epoch: {}...".format(e), end=" ")
        print("Discriminator loss on fakes: {} and on reals: {}".format(discriminator_loss_fake, discriminator_loss_true))
plt.figure(figsize=(16, 10))
for i in range(5*5):
    sample_data = generator.predict(tf.random.normal([1, 100]))
    sample_image = postprocess(sample_data)
    plt.subplot(5, 5, 1 + i)
    plt.axis('off')
    plt.imshow(sample_image, cmap='gray_r')
plt.show();
# conditional version


# DISCRIMINATOR
def define_discriminator(in_shape=(28,28,1), n_classes=10):
    
    # label input mapping to embedding
    label_input = tf.keras.Input(shape=(1,))
    label_layer = tf.keras.layers.Embedding(n_classes, 50)(label_input)
    label_layer = tf.keras.layers.Dense(in_shape[0] * in_shape[1])(label_layer)
    label_layer = tf.keras.layers.Reshape((in_shape[0], in_shape[1], in_shape[2]))(label_layer)
    
    # image input
    image_input = tf.keras.Input(shape=in_shape)
    
    # combine
    merged_layer = tf.keras.layers.Concatenate()([label_layer, image_input])
    merged_layer = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding="same")(merged_layer)
    merged_layer = tf.keras.layers.LeakyReLU(alpha=0.2)(merged_layer)
    merged_layer = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding="same")(merged_layer)
    merged_layer = tf.keras.layers.LeakyReLU(alpha=0.2)(merged_layer)
    merged_layer = tf.keras.layers.Flatten()(merged_layer)
    merged_layer = tf.keras.layers.Dropout(0.4)(merged_layer)
    
    out_layer = tf.keras.layers.Dense(1, activation="sigmoid")(merged_layer)
    model = tf.keras.Model([label_input, image_input], out_layer)
    optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model
    

# GENERATOR -- this is ultimately what we are interested in
def define_generator(n_classes=10):
    
    # label input mapping to an embedding space which can enrich each label with complex transformations
    label_input = tf.keras.Input(shape=(1,))
    label_layer = tf.keras.layers.Embedding(n_classes, 50)(label_input)
    label_layer = tf.keras.layers.Dense(7*7)(label_layer)
    label_layer = tf.keras.layers.Reshape((7,7,1))(label_layer)
    
    # random image generation
    random_input = tf.keras.Input(shape=(100,))
    random_layer = tf.keras.layers.Dense(7*7*128)(random_input)
    random_layer = tf.keras.layers.LeakyReLU(alpha=0.2)(random_layer)
    random_layer = tf.keras.layers.Reshape((7,7,128))(random_layer)
    
    # combined
    merged_layer = tf.keras.layers.Concatenate()([label_layer, random_layer])
    merged_layer = tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding="same")(merged_layer)
    merged_layer = tf.keras.layers.LeakyReLU(alpha=0.2)(merged_layer)
    merged_layer = tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding="same")(merged_layer)
    merged_layer = tf.keras.layers.LeakyReLU(alpha=0.2)(merged_layer)
    out_layer = tf.keras.layers.Conv2D(1, (7,7), activation="tanh", padding="same")(merged_layer)
    
    return tf.keras.Model([label_input, random_input], out_layer)


def define_gan(generator, discriminator):
    discriminator.trainable = False
    label, noise = generator.input
    generated_image = generator.output
    discriminator_decision = discriminator([label, generated_image])
    model = tf.keras.Model([label, noise], discriminator_decision)
    optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)
    return model
BATCH_SIZE=1000
EPOCHS=50

discriminator = define_discriminator()
generator = define_generator()
gan = define_gan(generator, discriminator)

# training
for e in range(EPOCHS):
    for images, labels in mnist_data.train.map(preprocess).batch(BATCH_SIZE):
        noise = tf.random.normal([BATCH_SIZE, 100])
        fake_images = generator.predict([labels, noise])
        discriminator_loss_fake, _ = discriminator.train_on_batch([labels, fake_images], tf.zeros(BATCH_SIZE))
        discriminator_loss_true, _ = discriminator.train_on_batch([labels, images], tf.ones(BATCH_SIZE))
        gan_loss = gan.train_on_batch([labels, noise], tf.ones(BATCH_SIZE))
    if (e % 10) == 0:
        print("Epoch: {}...".format(e), end=" ")
        print("Discriminator loss on fakes: {} and on reals: {}".format(discriminator_loss_fake, discriminator_loss_true))
l = 0
plt.figure(figsize=(16, 10))
for i in range(5*5):
    sample_data = generator.predict([np.array([l]), tf.random.normal([1, 100])])
    sample_image = postprocess(sample_data)
    l = l+1 if l < 9 else 0
    plt.subplot(5, 5, 1 + i)
    plt.axis('off')
    plt.imshow(sample_image, cmap='gray_r')
plt.show();
