import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import models, datasets, layers, optimizers, metrics, losses, utils
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
(train_values, train_labels), (test_values, test_labels) = datasets.reuters.load_data(num_words=15000)


print(f'{train_values.shape} Number of training records in reuters dataset')
print(f'{test_values.shape} Number of testing records in reuters dataset')
print('Values inside 1st record')
print(train_values[0])
print(f'Label of 1st record {train_labels[0]}')
# We have restricted number of words to 15000. Let check it
max([max(seq) for seq in train_values])
# word index from inbuild dataset
word2index = datasets.imdb.get_word_index()
index2word = {value: keyword for keyword, value in word2index.items()}
# Let check words present in both dictionary word2index and index2word
print(f'Awesome word index is {word2index.get("awesome")}')
print(f'At 1187 index which word is present {index2word.get(1187)}')
# (index - 3) because 0,1,2 index's are already assign to padding, start of sequence and unknown respectively
print('Actual sentence present inside 1st record')
print(' '.join([index2word.get(index - 3, '?') for index in train_values[0]]))
# Let's vectorize sentences with onehot encoding approach
def onehot_vectorization(sequences, num_words=15000):
    vectors = np.zeros((len(sequences), num_words))
    for i, seq in enumerate(sequences):
        vectors[i, seq] = 1.0
        
    return vectors
x_train = onehot_vectorization(train_values)
x_test  = onehot_vectorization(test_values)

print(f'Shape of train values are {x_train.shape} and test values are {x_test.shape}')
# Let's print original vector and onehot vector of 1st record
print(train_values[0])
print(x_train[0])
# Let's have look into the train and test labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print(y_train)
print(y_test)
# Let's first of all figure out how many class we have
print(f'Number of unique output categories we have in reuters is {len(np.unique(y_train))}')
# Now you have seen the output labels in above section and based on that we have to choose loss function. Let's talk about it more
# When you have values like [0, 1, 2, 3, 4, 5] in output labels it is suggest to use sparse_categorical_entropy as a loss function
# Or you have option of converting this output labels into onehot encoding just like we did for our input records
# We are going to use onehot version of our output records 

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

print(y_train.shape)
print(y_test.shape)
# Split dataset into train and validation making train data of 15000 and 10000 records respectively
random_shuf = np.arange(x_train.shape[0])
np.random.shuffle(random_shuf)

x_valid = x_train[random_shuf[:2000]]
y_valid = y_train[random_shuf[:2000]]

x_train = x_train[random_shuf[2000:]]
y_train = y_train[random_shuf[2000:]]

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(15000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.RMSprop(lr=0.001), metrics=metrics.categorical_accuracy)
history = model.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_valid, y_valid))
history_dict = history.history
print(history_dict.keys())
loss_values = history_dict.get('loss')
val_loss_values = history_dict.get('val_loss')

epochs = range(1, len(history_dict.get('loss')) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'o', label='Validtion loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.clf()

acc_values = history_dict.get('categorical_accuracy')
val_acc_values = history_dict.get('val_categorical_accuracy')

epochs = range(1, len(history_dict.get('categorical_accuracy')) + 1)

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'o', label='Validtion accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
# Try another network with more neurons then previously mentioned
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(15000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.RMSprop(lr=0.001), metrics=metrics.categorical_accuracy)
history = model.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_valid, y_valid))
loss_values = history_dict.get('loss')
val_loss_values = history_dict.get('val_loss')

epochs = range(1, len(history_dict.get('loss')) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'o', label='Validtion loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.clf()

acc_values = history_dict.get('categorical_accuracy')
val_acc_values = history_dict.get('val_categorical_accuracy')

epochs = range(1, len(history_dict.get('categorical_accuracy')) + 1)

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'o', label='Validtion accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
# By observing both graphs we can say that 8-9 epochs is more than enough for this architecture to perform best on both training and validation records
# Now train same network with 9 epochs
# By recompiling same network all the train weights will be lost and it will be retraining again from scratch

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(15000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.RMSprop(lr=0.001), metrics=metrics.categorical_accuracy)
history = model.fit(x_train, y_train, epochs=9, batch_size=512, validation_data=(x_valid, y_valid))
results = model.evaluate(x_test, y_test)
print(f'The loss on test records is {results[0]:.3f} and accuracy is {results[1]:.3f}')
predictions = model.predict(x_test)
print(predictions)
# Now to find each categories we have to use argsmax
print(predictions[0])
print('\n')
print(f'Maximum probability is {np.max(predictions[0]):.3f} at {np.argmax(predictions[0])} position')
