import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow.keras as keras

import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
imdb_reviews_word_ds = tfds.load("imdb_reviews", as_supervised=True)

imdb_reviews_word_ds_training = list(imdb_reviews_word_ds["train"])

imdb_reviews_word_ds_validation = list(imdb_reviews_word_ds["test"])
x_train_word_ds = []

y_train_word_ds = []



for text, label in imdb_reviews_word_ds_training:

    

    x_train_word_ds.append(text.numpy().decode("utf-8"))

    y_train_word_ds.append(label.numpy())

    



x_validation_word_ds = []

y_validation_word_ds = []



for text, label in imdb_reviews_word_ds_validation:

    

    x_validation_word_ds.append(text.numpy().decode("utf-8"))

    y_validation_word_ds.append(label.numpy())
tokenizer = keras.preprocessing.text.Tokenizer(num_words=12000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts=x_train_word_ds)
tokenizer.word_index
x_train_word_ds = tokenizer.texts_to_sequences(texts=x_train_word_ds)

x_validation_word_ds = tokenizer.texts_to_sequences(texts=x_validation_word_ds)
x_train_word_ds = keras.preprocessing.sequence.pad_sequences(sequences=x_train_word_ds,

                                                             maxlen=128,

                                                             padding="post",

                                                             truncating="post")



x_validation_word_ds = keras.preprocessing.sequence.pad_sequences(sequences=x_validation_word_ds,

                                                                  maxlen=128,

                                                                  padding="post",

                                                                  truncating="post")
# word dataset for training

temp_dict = {"Text (Word)": list(x_train_word_ds),

             "Label (int)": y_train_word_ds}



pd.DataFrame.from_dict(temp_dict)
# word dataset for validation

temp_dict = {"Text (Word)": list(x_validation_word_ds),

             "Label (int)": y_validation_word_ds}



pd.DataFrame.from_dict(temp_dict)
x_train_word_ds = np.array(x_train_word_ds)

y_train_word_ds = np.array(y_train_word_ds)

print("Type of word dataset for training (Text): ", type(x_train_word_ds))

print("Type of word dataset for training (Label): ", type(y_train_word_ds))



print()



x_validation_word_ds = np.array(x_validation_word_ds)

y_validation_word_ds = np.array(y_validation_word_ds)

print("Type of word dataset for validation (Text): ", type(x_validation_word_ds))

print("Type of word dataset for validation (Label): ", type(y_validation_word_ds))
print("Shape of word dataset for training (Text): ", x_train_word_ds.shape)

print("Shape of word dataset for training (Label): ", y_train_word_ds.shape)

print("Shape of word dataset for validation (Text): ", x_validation_word_ds.shape)

print("Shape of word dataset for validation (Label): ", y_validation_word_ds.shape)
imdb_reviews_subword_ds, info = tfds.load("imdb_reviews/subwords8k", as_supervised=True, with_info=True)

imdb_reviews_subword_ds_training = list(imdb_reviews_subword_ds["train"])

imdb_reviews_subword_ds_validation = list(imdb_reviews_subword_ds["test"])
x_train_subword_ds = []

y_train_subword_ds = []



for text, label in imdb_reviews_subword_ds_training:

    

    x_train_subword_ds.append(text.numpy())

    y_train_subword_ds.append(label.numpy())

    



x_validation_subword_ds = []

y_validation_subword_ds = []



for text, label in imdb_reviews_subword_ds_validation:

    

    x_validation_subword_ds.append(text.numpy())

    y_validation_subword_ds.append(label.numpy())
x_train_subword_ds = keras.preprocessing.sequence.pad_sequences(sequences=x_train_subword_ds,

                                                                maxlen=200,

                                                                padding="post",

                                                                truncating="post")



x_validation_subword_ds = keras.preprocessing.sequence.pad_sequences(sequences=x_validation_subword_ds,

                                                                     maxlen=200,

                                                                     padding="post",

                                                                     truncating="post")
# subword dataset for training

temp_dict = {"Text (Subword)": list(x_train_subword_ds),

             "Label (int)": y_train_subword_ds}



pd.DataFrame.from_dict(temp_dict)
# subword dataset for validation

temp_dict = {"Text (Subword)": list(x_validation_subword_ds),

             "Label (int)": y_validation_subword_ds}



pd.DataFrame.from_dict(temp_dict)
x_train_subword_ds = np.array(x_train_subword_ds)

y_train_subword_ds = np.array(y_train_subword_ds)

print("Type of subword dataset for training (Text): ", type(x_train_subword_ds))

print("Type of subword dataset for training (Label): ", type(y_train_subword_ds))



print()



x_validation_subword_ds = np.array(x_validation_subword_ds)

y_validation_subword_ds = np.array(y_validation_subword_ds)

print("Type of subword dataset for validation (Text): ", type(x_validation_subword_ds))

print("Type of subword dataset for validation (Label): ", type(y_validation_subword_ds))
print("Shape of subword dataset for training (Text): ", x_train_subword_ds.shape)

print("Shape of subword dataset for training (Label): ", y_train_subword_ds.shape)

print("Shape of subword dataset for validation (Text): ", x_validation_subword_ds.shape)

print("Shape of subword dataset for validation (Label): ", y_validation_subword_ds.shape)
model_word = keras.models.Sequential()

model_word.add(keras.layers.Embedding(input_dim=12000, output_dim=32, input_length=128))

model_word.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True)))

model_word.add(keras.layers.Bidirectional(keras.layers.LSTM(units=64)))

model_word.add(keras.layers.Dense(units=16, activation="relu"))

model_word.add(keras.layers.Dense(units=1, activation="sigmoid"))
model_word.summary()
model_word.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=["acc"])
class CustomCallback(keras.callbacks.Callback):

    

    def on_epoch_end(self, epochs, logs):

        if logs["acc"] >= 0.99:

            self.model.stop_training = True



my_callback = CustomCallback()
model_subword = keras.models.Sequential()

model_subword.add(keras.layers.Embedding(input_dim=info.features["text"].encoder.vocab_size, output_dim=32, input_length=200))

model_subword.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True)))

model_subword.add(keras.layers.Bidirectional(keras.layers.LSTM(units=64)))

model_subword.add(keras.layers.Dense(units=16, activation="relu"))

model_subword.add(keras.layers.Dense(units=1, activation="sigmoid"))
model_subword.summary()
model_subword.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=["acc"])
pass
history_word = model_word.fit(x=x_train_word_ds,

               y=y_train_word_ds,

               batch_size=32,

               epochs=50,

               callbacks=my_callback,

               validation_data=(x_validation_word_ds, y_validation_word_ds))
histroy_subword = model_subword.fit(x=x_train_subword_ds,

                                    y=y_train_subword_ds,

                                    batch_size=32,

                                    epochs=50,

                                    callbacks=my_callback,

                                    validation_data=(x_validation_subword_ds, y_validation_subword_ds))
training_acc = history_word.history["acc"]

validation_acc = history_word.history["val_acc"]

epochs = list(range(len(validation_acc)))



plt.plot(epochs, training_acc, color="blue", label="Training Acc")

plt.plot(epochs, validation_acc, color="red", label="Validation Acc")

plt.legend()

plt.show()
training_acc = histroy_subword.history["acc"]

validation_acc = histroy_subword.history["val_acc"]

epochs = list(range(len(validation_acc)))



plt.plot(epochs, training_acc, color="blue", label="Training Acc")

plt.plot(epochs, validation_acc, color="red", label="Validation Acc")

plt.legend()

plt.show()