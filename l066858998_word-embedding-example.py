import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow.keras as keras

import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
dataset.head()
dataset.tail()
dataset.shape
dataset = dataset.values
training_data_ratio = 0.8

np.random.shuffle(dataset)



training_dataset = dataset[:int(dataset.shape[0] * training_data_ratio), :]

validation_dataset = dataset[int(dataset.shape[0] * training_data_ratio):, :]
print("The shape of training dataset: ", training_dataset.shape)

print("The shape of validation dataset: ", validation_dataset.shape)
# create a tokenizer

tokenizer = keras.preprocessing.text.Tokenizer(num_words=12000, oov_token="<OOV>")
# fit tokenizer on training sentences

tokenizer.fit_on_texts(training_dataset[:, 0])
# show the word - value pair

tokenizer.word_index
# tokenize training sentences and validation sentences

x_train = tokenizer.texts_to_sequences(training_dataset[:, 0])

x_validation = tokenizer.texts_to_sequences(validation_dataset[:, 0])
# length of sequences in training set are different, so are sentences in validation set

print("The length of sentence in training set: ", len(x_train[0]))

print("The length of sentence in training set: ", len(x_train[100]))



print("The length of sentence in validation set: ", len(x_validation[0]))

print("The length of sentence in validation set: ", len(x_validation[100]))
# pad the sentence to make them same length

x_train = keras.preprocessing.sequence.pad_sequences(sequences=x_train, 

                                           maxlen=256,

                                           padding="post",

                                           truncating="post")



x_validation = keras.preprocessing.sequence.pad_sequences(sequences=x_validation, 

                                                          maxlen=256,

                                                          padding="post",

                                                          truncating="post")
print("The length of sentence in training set: ", len(x_train[0]))

print("The length of sentence in training set: ", len(x_train[100]))



print("The length of sentence in validation set: ", len(x_validation[0]))

print("The length of sentence in validation set: ", len(x_validation[100]))
# convert training labels and validation labels to one-of-two encoding form

y_train = training_dataset[: ,1]

y_validation = validation_dataset[:, 1]



print("Training Labels: ", y_train)

print("Validation Labels: ", y_validation)
# convert string two integer

mask_pos = (y_train == "positive")

mask_neg = (y_train == "negative")

y_train[mask_pos] = 1

y_train[mask_neg] = 0





mask_pos = (y_validation == "positive")

mask_neg = (y_validation == "negative")

y_validation[mask_pos] = 1

y_validation[mask_neg] = 0
# one-of-two encoding form

y_train = keras.utils.to_categorical(y=y_train, num_classes=2)

y_validation = keras.utils.to_categorical(y=y_validation, num_classes=2)
# have a look on part on training data

print("Part of Training Data: ")

print()

print("Sentences: ")

print(x_train[0:10])

print()

print("Labels: ")

print(y_train[0:10])
# have a look on part on validation data

print("Part of Validation Data: ")

print()

print("Sentences: ")

print(x_validation[0:10])

print()

print("Labels: ")

print(y_validation[0:10])
inputs = keras.layers.Input(shape=(256, ))

x = keras.layers.Embedding(input_dim=12000, output_dim=48, input_length=256)(inputs)

x = keras.layers.Flatten()(x)

x = keras.layers.Dense(units=24, activation="relu")(x)

x = keras.layers.Dense(units=12, activation="relu")(x)

outputs = keras.layers.Dense(units=2, activation="softmax")(x)



model = keras.models.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=["acc"])
class CustomCallback(keras.callbacks.Callback):

    

    def on_epoch_end(self, epoch, logs):

        if logs["acc"] >= 0.99:

            self.model.stop_training = True

    

custom_callback = CustomCallback()
history = model.fit(x=x_train,

          y=y_train,

          batch_size=32,

          epochs=20,

          callbacks=custom_callback,

          validation_data=(x_validation, y_validation))
training_acc = history.history["acc"]

validation_acc = history.history["val_acc"]

epoch = list(range(len(training_acc)))



plt.plot(epoch, training_acc, "b", label="Training Acc")

plt.plot(epoch, validation_acc, "r", label="Validation Acc")



plt.legend()

plt.show()
# get the embedding layer in model

embedding_layer = model.layers[1]



# get the weight of embedding layer

embedding_matrix = embedding_layer.get_weights()[0]
# each row in embedding matrix represents embedding vector of word

embedding_matrix.shape