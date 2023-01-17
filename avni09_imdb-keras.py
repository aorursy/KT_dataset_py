from keras.datasets import imdb

from keras import layers, models, optimizers

import numpy as np

import matplotlib.pyplot as plt

def vectorize_sequence(sequences, dimension=1000):

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        results[i,sequence] = 1.

    return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 1000)
# print("Max index in train data: {max([max(sequence) for sequence in train_data])}")

# print("Train label: train_labels[0] = {train_labels[0]})

# print({len(train_data[0])})
word_index = imdb.get_word_index() # maps words to index, commonly called as w_to_i

reverse_word_index = dict( [ (value, key) for (key, value) in word_index.items()] ) # this is just opposite, i_to_w

decode_review = ' '.join( [reverse_word_index.get(i - 3, '?') for i in train_data[0] ])
# print(f"Encoded Review: \n {train_data[0]} \n")

print(decode_review)

x_train = vectorize_sequence(train_data) # mapping one hot encoding to a particular index

x_test = vectorize_sequence(test_data) # mapping one hot encoding to a particular index
y_train = np.asarray(train_labels).astype('float32')

y_test = np.asarray(test_labels).astype('float32')
# print("A data point appears like" (x_train[0]))

# print("Actual appears as: {y_train[0]}")
# Validation separation



x_val = x_train[:1000]

partial_x_train = x_train[1000:]



y_val = y_train[:1000]

partial_y_train = y_train[1000:]
# Model definition



model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(1000,), name="Input_Layer"))

model.add(layers.Dense(16, activation='relu', name="Hidden_Layer_1"))

model.add(layers.Dense(1, activation='sigmoid', name="Output_Layer"))

print(model.summary())

# Compiling the model



model.compile(optimizer=optimizers.RMSprop(lr = 0.001),

              loss='binary_crossentropy',

              metrics=['accuracy'])

history = model.fit(partial_x_train,

                    partial_y_train,

                    epochs = 4,

                    batch_size = 128,

                    validation_data = (x_val, y_val))

history_dict = history.history

acc = history_dict["acc"]

val_acc = history_dict["val_acc"]

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

print (loss_values)
epochs = range(1, len(acc) + 1)
print (epochs)
import matplotlib.pyplot as plt

plt.figure(1)

plt.subplot(211)

plt.plot(epochs, loss_values, 'bo', label='Training Loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.ylabel('Loss')

plt.legend()

plt.show()
plt.subplot(212)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label = 'Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
# Running Predict

test = x_test[1].reshape(1, -1)

print(f"Model's Prediction: {model.predict(test)[0]}")

print(f"Actual: {y_test[1]}")