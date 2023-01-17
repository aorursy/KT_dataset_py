# Importing required modules



import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.callbacks import LambdaCallback 

import tensorflow.keras.layers as L

from tensorflow.keras.datasets import mnist, cifar10



import numpy as np

import matplotlib.pyplot as plt

from random import randint

from tqdm import tqdm



print(f'Tensorflow version: {tf.__version__}')
# Utility functions

def print_shapes(x_train, x_test, y_train, y_test):

  print(f"x_train: {x_train.shape}\n"\

      f"x_test: {x_test.shape}\n"\

      f"y_train: {y_train.shape}\n"\

      f"y_test: {y_test.shape}\n")
# loading the dataset

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print_shapes(x_train, x_test, y_train, y_test)
# Preprocessing images and labels

height, width, channels = 32, 32, 3

nb_classes = 10 

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',

               'horse', 'ship', 'truck']



x_train = x_train / 255

x_test = x_test / 255



x_train = x_train.reshape((-1, height, width, channels))

x_test = x_test.reshape((-1, height, width, channels))



y_train = tf.keras.utils.to_categorical(y_train, nb_classes)

y_test = tf.keras.utils.to_categorical(y_test, nb_classes)



print_shapes(x_train, x_test, y_train, y_test)
# Building a simple CNN model

model = Sequential()



model.add(L.Conv2D(128, kernel_size=(3, 3),

                 padding='same', activation='relu', 

                 input_shape=(height, width, channels)))

model.add(L.Dropout(0.3))



model.add(L.Conv2D(64, kernel_size=(3, 3),

                 padding='same', activation='relu', 

                 input_shape=(height, width, channels)))

model.add(L.Dropout(0.3))



model.add(L.Conv2D(64, kernel_size=(3, 3),

                 padding='same', activation='relu'))

model.add(L.Dropout(0.3))

model.add(L.MaxPooling2D(pool_size=(2, 2)))



model.add(L.Conv2D(64, kernel_size=(3, 3),

                 padding='same', activation='relu'))

model.add(L.MaxPooling2D(pool_size=(2, 2)))



model.add(L.Dropout(0.3))

model.add(L.Flatten())

model.add(L.Dense(32))

model.add(L.Dropout(0.2))

model.add(L.Dense(nb_classes, activation='softmax'))



model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])



model.summary()
# Training the model

history = model.fit(x_train, y_train,

                    batch_size=32,

                    epochs=20,

                    validation_data=(x_test, y_test))
# plotting loss

plt.plot(history.history['loss'], label="loss")

plt.plot(history.history['val_loss'], label="val_loss")

plt.legend()

plt.show()
# plotting accuracy

plt.plot(history.history['accuracy'], label="accuracy")

plt.plot(history.history['val_accuracy'], label="val_accuracy")

plt.legend()

plt.show()
# Function to calculate adversary noise

def generate_adversary(image, label):

  image = tf.cast(image, tf.float32)



  with tf.GradientTape() as tape:

    tape.watch(image)

    prediction = model(image)

    loss = tf.keras.losses.MSE(label, prediction)

  gradient = tape.gradient(loss, image)

  sign_grad = tf.sign(gradient)



  return sign_grad
# Selecting random image for testing

rand_idx = randint(0,49999)

image = x_train[rand_idx].reshape((1, height, width, channels))

label = y_train[rand_idx]



print(f'Prediction from CNN: {label_names[np.where(label==1)[0][0]]}')

plt.figure(figsize=(3,3))

plt.imshow(image.reshape((height, width, channels)))

plt.show()
# Adding the adversary noise to image

perturbations = generate_adversary(image,label).numpy()

adversarial = image + (perturbations * 0.05)
# Comparing both images 

fig, (ax1,ax2) = plt.subplots(1, 2, sharey=True)

ax1.imshow(image.reshape(height,width, channels))

ax1.set_title("Original Image")

ax2.imshow(adversarial.reshape(height,width, channels))

ax2.set_title("Image with Adversary")

plt.show()
# Comparing predictions

print(f'Normal Image Prediction: {label_names[model.predict(image).argmax()]}')

print(f"Adversary Prediction: {label_names[model.predict(adversarial).argmax()]}")
# Function to generate batch of images with adversary

def adversary_generator(batch_size):

  while True:

    images = []

    labels = []

    for batch in range(batch_size):

      N = randint(0, 49999)

      label = y_train[N]

      image = x_train[N].reshape((1,height, width, channels))



      perturbations = generate_adversary(image, label).numpy()

      adversarial = image + (perturbations * 0.1)



      images.append(adversarial)

      labels.append(label)



      if batch%1000 == 0:

        print(f"{batch} images generated")



    images = np.asarray(images).reshape((batch_size, height, width, channels))

    labels = np.asarray(labels)



    yield images, labels
# Testing model accuracy on adversarial examples

x_adversarial, y_adversarial = next(adversary_generator(10000))

ad_acc = model.evaluate(x_adversarial, y_adversarial, verbose=0)

print(f"Accuracy on Adversarial Examples: {ad_acc[1]*100}")