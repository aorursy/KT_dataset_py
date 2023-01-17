import keras

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from kerastuner import RandomSearch

from kerastuner.engine.hyperparameters import HyperParameters
fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
plt.figure(figsize=(10,10))

for i in range(20):

    plt.subplot(4,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(X_train[i], cmap = plt.cm.binary)

plt.show()
X_train.shape, X_test.shape
classes = {0 : "T-shirt/top", 1 : "Trouser", 2 : "Pullover", 3 : "Dress", 4 : "Coat",

           5 : "Sandal",      6 : "Shirt",   7 : "Sneaker",  8 : "Bag",   9 : "Ankle Boot"}
X_train = X_train / 255.0

X_test  = X_test  / 255.0
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train.shape, X_test.shape
def build_model(params):

  model = keras.Sequential([

                            keras.layers.Conv2D(filters = params.Int('conv_1_filter', min_value = 32, max_value = 128, step = 16),

                                                kernel_size = params.Choice('conv_1_kernel', values = [3,5]),

                                                activation = 'relu',

                                                input_shape = (28,28,1)

                                               ),

                            keras.layers.Conv2D(filters = params.Int('conv_2_filter', min_value = 32, max_value = 64, step = 16),

                                                kernel_size = params.Choice('conv_2_kernel', values = [3,5]),

                                                activation = 'relu'

                                                ),

                            keras.layers.Flatten(),

                            keras.layers.Dense(units = params.Int('dense_1_units', min_value = 32, max_value = 64, step = 16),

                                               activation = 'relu'

                                              ),

                            keras.layers.Dense(10, activation='softmax')

                          ])

  

  model.compile(optimizer=keras.optimizers.Adam(params.Choice('learning_rate', values = [1e-2, 1e-3])),

                loss = 'sparse_categorical_crossentropy',

                metrics = ['accuracy']

               )



  return model  
tuner = RandomSearch(build_model, 

                     objective='val_accuracy',

                     max_trials = 5,

                     directory = 'output',

                     project_name = 'Fashion Mnist')
tuner.search_space_summary()
tuner.search(X_train, y_train, epochs=3, validation_split=0.1)
model = tuner.get_best_models(num_models=1)[0]
model.summary()
model.fit(X_train, y_train, epochs=10, initial_epoch=3, validation_data=(X_test, y_test), validation_split=0.1)
preds = model.predict(X_test)
X_test = X_test.reshape(X_test.shape[0], 28, 28)

X_test.shape
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,7,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(X_test[i], cmap = plt.cm.binary)

    plt.xlabel(classes[y_test[i]])

    plt.title(classes[np.argmax(preds[i])])

plt.show()
def plot_image(prediction, img):

    plt.xticks([])

    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = classes[np.argmax(prediction)]

    plt.xlabel("Predicted:{} ".format(predicted_label,

               

               ),

                color="red")
def plot_value_array(prediction):

    plt.xticks(range(10))

    plt.yticks([])

    thisplot = plt.bar(range(10), prediction)

    plt.ylim([0,1])

    predicted_label = np.argmax(prediction)

    thisplot[predicted_label].set_color('salmon')



plt.figure(figsize=(8,12))

for i in range(5):

    plt.subplot(5, 2, 2*i+1)

    #plt.xticks([])

    #plt.yticks([])

    plot_image(preds[i], X_test[i])

    #plt.imshow( X_test[i], cmap = plt.cm.binary)

    plt.subplot(5, 2, 2*i+2)

    plot_value_array(preds[i])

plt.show()  