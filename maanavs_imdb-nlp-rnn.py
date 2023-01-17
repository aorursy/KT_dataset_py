import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

import numpy as np

train_data, test_data = imdb['train'], imdb['test']
training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s,l in train_data:
  training_sentences.append(str(s.numpy()))
  training_labels.append(l.numpy())

for s,l in test_data:
  testing_sentences.append(str(s.numpy()))
  testing_labels.append(l.numpy())
print(len(training_labels))
print(len(testing_labels))

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)
vocab_size = 10000
embedding_dim = 16
max_length = 140
trunc_type='post'
oov_tok='<OOV>'

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, truncating=trunc_type)
import random as rnd
randIndex = rnd.randrange(0, len(testing_sentences))
print(randIndex)
print(testing_sentences[randIndex])
print(testing_labels[randIndex])
import tensorflow as tf
model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
                             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
                             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                             #tf.keras.layers.Conv1D(16, activation='relu'),
                             #tf.keras.layers.Dense(512, activation = 'relu'),
                             tf.keras.layers.Dense(128, activation='relu'),
                             #tf.keras.layers.Dropout(.2),
                             tf.keras.layers.Dense(1, activation = 'sigmoid')])

model.compile(loss ='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(padded, training_labels_final, epochs=10, validation_data=(testing_padded, testing_labels_final))
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
import random as rnd
randIndex = rnd.randrange(0, len(testing_sentences))
#print(randIndex)
#print(testing_sentences[randIndex])
#print(testing_labels[randIndex])

testStringSequence = tokenizer.texts_to_sequences([testing_sentences[randIndex]])
paddedSequence = pad_sequences(testStringSequence, maxlen=max_length, truncating=trunc_type)

prediction = model.predict(paddedSequence)
for review in range(len(prediction)):
  if prediction[review][0] >= .5:
    pred = "Positive Rating"
  else:
    pred = "Negative Rating"
  print("{testStr} : {preds}".format(testStr = testing_sentences[randIndex], preds=pred))
  print(prediction[review][0])
  print(testing_labels[randIndex])
