import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

tf.enable_eager_execution()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


img_input = tf.keras.layers.Input(shape=(28,28,1))

x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(img_input)

x = tf.keras.layers.MaxPooling2D((2,2))(x)



x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.Dense(512, activation='relu')(x)

output = tf.keras.layers.Dense(10, activation='softmax')(x)





model = tf.keras.Model(img_input, output)
model.summary()
x_train = np.expand_dims(x_train, axis = 3)

x_train.shape
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2)
x_test = np.expand_dims(x_test,3)

model.evaluate(x_test,y_test)
model.layers
a = np.array(([1,2],[4,5]))

display(a)

print(a.shape)
b = np.expand_dims(a, axis =3)

b.shape
model.layers[1].output
my_model = tf.keras.Model(img_input, [model.layers[1].output, model.layers[2].output])
input = np.expand_dims(x_train[10], axis=0)

pred = my_model.predict(input)
from tensorflow.keras.preprocessing.image import img_to_array, load_img

successive_outputs = [layer.output for layer in model.layers[1:]]

visualization_model = tf.keras.Model(img_input, successive_outputs)

x = input

# Let's run our image through our network, thus obtaining all

# intermediate representations for this image.

successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot

layer_names = [layer.name for layer in model.layers]
# Now let's display our representations

for layer_name, feature_map in zip(layer_names, successive_feature_maps):

  if len(feature_map.shape) == 4:

    # Just do this for the conv / maxpool layers, not the fully-connected layers

    n_features = feature_map.shape[-1]  # number of features in feature map

    # The feature map has shape (1, size, size, n_features)

    size = feature_map.shape[1]

    # We will tile our images in this matrix

    display_grid = np.zeros((size, size * n_features))

    for i in range(n_features):

      # Postprocess the feature to make it visually palatable

      x = feature_map[0, :, :, i]

      x -= x.mean()

      x /= x.std()

      x *= 64

      x += 128

      x = np.clip(x, 0, 255).astype('uint8')

      # We'll tile each filter into this big horizontal grid

      display_grid[:, i * size : (i + 1) * size] = x

    # Display the grid

    scale = 40. / n_features

    plt.figure(figsize=(scale * n_features, scale))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')