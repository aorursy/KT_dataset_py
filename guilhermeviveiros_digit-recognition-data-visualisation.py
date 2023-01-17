import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np
def show_history(history):

    print(history.history.keys())



    # summarize history for accuracy

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'val'], loc='upper left')

    plt.show()

    # summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'val'], loc='upper left')

    plt.show()
mnist = tf.keras.datasets.mnist



(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0



x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))

x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

print(x_train.shape)

model = tf.keras.models.Sequential([

  tf.keras.layers.Flatten(input_shape=(28, 28,1)),

  tf.keras.layers.Dense(256, activation='relu'),

  tf.keras.layers.Dense(256, activation='relu'),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(10, activation='softmax')

])



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



print(model.summary())
history = model.fit(x_train, y_train, batch_size=32, epochs=2, validation_split=0.2, verbose=0)
show_history(history)
evaluate = model.evaluate(x_test,  y_test, verbose=0)

print("Accuracy test of = " + str(evaluate[1]))
import keras

from keras.models import Sequential, Model

from keras.layers import Conv2D, Flatten, Dense, BatchNormalization





modelConv = Sequential()

modelConv.add(Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]))

modelConv.add(BatchNormalization())

modelConv.add(Conv2D(32, (3, 3)))

modelConv.add(BatchNormalization())

modelConv.add(Flatten())

modelConv.add(Dense(32, activation='relu'))

modelConv.add(Dense(10, activation='softmax'))





modelConv.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



print(modelConv.summary())
from tensorflow.keras.callbacks import ModelCheckpoint



checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", 

                               monitor = 'val_accuracy',

                               verbose=1, 

                               save_best_only=True)



history = modelConv.fit(x_train, y_train, 

                        batch_size=32, 

                        epochs=10, 

                        #callbacks=[checkpointer],

                        validation_split=0.2)
show_history(history)
modelConv.evaluate(x_test,  y_test, verbose=2)[1]
print(modelConv.layers[0])



print(modelConv.layers)

layer_outputs = [layer.output for layer in modelConv.layers[:12]] # Extracts the outputs of the top 12 layers

activation_model = tf.keras.models.Model(inputs=modelConv.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
activations = activation_model.predict(x_test[:1])

first_layer_activation = activations[0]

print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
layer_names = []

for layer in modelConv.layers[:12]:

    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

    

images_per_row = 16



for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps

    n_features = layer_activation.shape[-1] # Number of features in the feature map

    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).

    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix

    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols): # Tiles each filter into a big horizontal grid

        for row in range(images_per_row):

            channel_image = layer_activation[0,

                                             :, :,

                                             col * images_per_row + row]

            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable

            channel_image /= channel_image.std()

            channel_image *= 64

            channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size : (col + 1) * size, # Displays the grid

                         row * size : (row + 1) * size] = channel_image

    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1],

                        scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')
