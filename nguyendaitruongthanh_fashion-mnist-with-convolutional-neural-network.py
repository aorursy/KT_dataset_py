import tensorflow as ft

import matplotlib.pyplot as plt



from tensorflow import keras

from keras import layers

from keras.datasets import fashion_mnist



# load the fashion mnist dataset from Keras API



(train_images_full, train_labels_full), (test_images, test_labels) = fashion_mnist.load_data()
# callback when the model reaches 99% accuracy



class tfCallback(keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if logs.get("accuracy")>=0.95:

      print("\nReached 95% accuracy!")

      self.model.stop_training = True 



callbacks=tfCallback()



# preprocess the the dataset



train_images_full = train_images_full.reshape(60000, 28, 28, 1)

train_images_full = train_images_full / 255.0

test_images = test_images.reshape(10000, 28, 28, 1)

test_images = test_images / 255.0
# create a validation set with 5000 examples from the training set



validation_images, train_images = train_images_full[:5000], train_images_full[5000:]

validation_labels, train_labels = train_labels_full[:5000], train_labels_full[5000:]



print(validation_images.shape)

print(train_images.shape)
# build a deep learning model with 2 CNN layers and a densely connected layer



model = keras.models.Sequential([layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),

                                 layers.MaxPooling2D(2,2),

                                 layers.Conv2D(32, (3,3), activation="relu"),

                                 layers.MaxPooling2D(2,2),

                                 layers.Dropout(0.2),

                                 layers.Flatten(),

                                 layers.Dense(128, activation="relu"),

                                 layers.Dense(10, activation="softmax")])



# use adam for model optimization



model.compile(loss="sparse_categorical_crossentropy", 

              optimizer="adam", 

              metrics=["accuracy"])



history = model.fit(train_images, 

                    train_labels, 

                    epochs=50,

                    validation_data=(validation_images, validation_labels),

                    callbacks=callbacks)
history_dict = history.history

history_dict.keys()
# plot the accuracy and loss



accuracy = history.history["accuracy"]

val_accuracy = history.history["val_accuracy"]

epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, "b-", label="Training accuracy")

plt.plot(epochs, val_accuracy, "bo", label="Validation accuracy")

plt.title("Training and validation accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.show()



loss = history.history["loss"]

val_loss = history.history["val_loss"]

plt.plot(epochs, loss, "r-", label="Training loss")

plt.plot(epochs, val_loss, "ro", label="Validation loss")

plt.title("Training and validation loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
# evaluate the model on test set



model.evaluate(test_images, test_labels)
prediction = model.predict(test_images)



print(prediction[6])

print(test_labels[6])