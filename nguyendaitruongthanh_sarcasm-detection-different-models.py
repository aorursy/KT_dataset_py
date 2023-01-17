import json 



with open('../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json','r') as f:

    

    data = f.read()

    data = "[" + data.replace("}", "},", data.count("}")-1) + "]"

    datastore = json.loads(data)
headlines = []

labels = []



for item in datastore:

    headlines.append(item["headline"])

    labels.append(item["is_sarcastic"])
import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np



from tensorflow import keras

from keras import layers

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
# configure important hyper-parameters 



vocab_size = 2000

max_len = 100

embedding_dim = 32

oov_tok = "<OOV>"

padding_type = "post"

trunc_type = "post"

training_size = 20000
# split the data into sets for training and testing



train_data, test_data = headlines[:training_size], headlines[training_size:]

train_labels, test_labels = labels[:training_size], labels[training_size:]
# tokenize the train_data and test_data



tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(train_data)



word_index = tokenizer.word_index



train_sequences = tokenizer.texts_to_sequences(train_data)

train_padded = pad_sequences(train_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)



test_sequences = tokenizer.texts_to_sequences(test_data)

test_padded = pad_sequences(test_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)



train_padded = np.array(train_padded)

train_labels = np.array(train_labels)



test_padded = np.array(test_padded)

test_labels = np.array(test_labels)
# define a model (version 1 - 1 bidirectional LSTM)



model_ver1 = keras.Sequential([layers.Embedding(vocab_size, embedding_dim, input_length=max_len),

                          layers.Bidirectional(layers.LSTM(16, return_sequences=True)),

                          layers.Dense(16, activation="relu"),

                          layers.Dense(1, activation="sigmoid")])

                         

model_ver1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

                         

model_ver1.summary()

keras.utils.plot_model(model_ver1)
# train the model



num_epochs = 20



history_1 = model_ver1.fit(train_padded, train_labels,

                   epochs=num_epochs,

                   validation_data=(test_padded, test_labels),

                   verbose=1)
# plot accuracy and loss



acc = history_1.history["accuracy"]

val_acc = history_1.history["val_accuracy"]

loss = history_1.history["loss"]

val_loss = history_1.history["val_loss"]



epochs = range(1, len(acc) + 1)



# accuracy



plt.plot(epochs, acc, "b", label="Training accuracy")

plt.plot(epochs, val_acc, "b--", label="Validation accuracy")

plt.title("Training and validation accuracy")

plt.legend()

plt.show()



# loss



plt.plot(epochs, loss, "r", label="Training loss")

plt.plot(epochs, val_loss, "r--", label="Validation loss")

plt.title("Training and validation loss")

plt.legend()

plt.show()
# define a model (version 2)



model_ver2 = keras.Sequential([layers.Embedding(vocab_size, embedding_dim, input_length=max_len),

                               layers.Dropout(0.4),

                               layers.Conv1D(64, 5, activation="relu"),

                               layers.MaxPooling1D(pool_size=4),

                               layers.LSTM(128),

                               layers.Dense(1, activation="sigmoid")])

                         

model_ver2.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

                         

model_ver2.summary()

keras.utils.plot_model(model_ver2)
# train the second model



num_epochs = 20



history_2 = model_ver2.fit(train_padded, train_labels,

                   epochs=num_epochs,

                   validation_data=(test_padded, test_labels),

                   verbose=1)
# plot accuracy and loss of the second model



acc = history_2.history["accuracy"]

val_acc = history_2.history["val_accuracy"]

loss = history_2.history["loss"]

val_loss = history_2.history["val_loss"]



epochs = range(1, len(acc) + 1)



# accuracy



plt.plot(epochs, acc, "b", label="Training accuracy")

plt.plot(epochs, val_acc, "b--", label="Validation accuracy")

plt.title("Training and validation accuracy")

plt.legend()

plt.show()



# loss



plt.plot(epochs, loss, "r", label="Training loss")

plt.plot(epochs, val_loss, "r--", label="Validation loss")

plt.title("Training and validation loss")

plt.legend()

plt.show()