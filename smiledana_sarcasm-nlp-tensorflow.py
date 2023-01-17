import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
print(os.listdir("../input/sarcasm"))
base_dir = "../input/sarcasm/"
with open(base_dir + 'train-balanced-sarcasm.csv') as f:
      data = pd.read_csv(f)
print(type(data))
print(data[1:3])
import numpy as np
vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

# load the data
with open('/tmp/sarcasm.json', 'r') as f:
      data = json.load(f)
print(type(data))
print(data[1:3])
sentences = []
labels = []
for item in data:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    
train_sentences = sentences[:training_size]
valid_sentences = sentences[training_size:]
train_labels = labels[:training_size]
valid_labels = labels[training_size:]

#sentences processing
tokenize = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenize.fit_on_texts(sentences)
word_index = tokenize.word_index

train_seq = tokenize.texts_to_sequences(train_sentences)
valid_seq = tokenize.texts_to_sequences(valid_sentences)

train_padded = pad_sequences(train_seq, truncating=trunc_type, padding=padding_type, maxlen=max_length)
valid_padded = pad_sequences(valid_seq, truncating=trunc_type, padding=padding_type, maxlen=max_length)

train_padded = np.array(train_padded)
valid_padded = np.array(valid_padded)
print(train_padded.shape)
print(valid_padded.shape)

#labels processing
train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)
print(train_labels.shape)
print(valid_labels.shape)
import tensorflow as tf
model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences = True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
])
 

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.90):
        print("\nReached 90% accuracy so cancelling training!")
        self.model.stop_training = True
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks = myCallback()
history = model.fit(train_padded, train_labels,
                        validation_data=(valid_padded, valid_labels),
                        epochs=100,
                        verbose=1,callbacks=[callbacks])
import matplotlib.pyplot as plt
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')