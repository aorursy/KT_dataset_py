import numpy as np
import pandas as pd


GLOVE_DIR = '../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt'
DATA_DIR = '../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv'

df = pd.read_csv(DATA_DIR)
df.replace(['positive', 'negative'], [1, 0], inplace=True)
df.head()
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Data preparing

data = list(df.review)
labels = list(df.sentiment)

tokenizer = Tokenizer(num_words=10000) # Only 10000 most met words
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)

data = pad_sequences(sequences, maxlen=100) # Only 100 words in one review
labels = np.array(labels)


train_data = data[:30000]
train_labels = labels[:30000]

validation_data = data[30000:40000]
validation_labels = labels[30000:40000]

test_data = data[40000:]
test_labels = labels[40000:]
embedding_index = {}
with open(GLOVE_DIR) as file: # Preparing glove
    for line in file:
        values = line.split()
        key = values[0]
        value = values[1:]
        embedding_index[key] = np.array(value, dtype='float32')
        
        
embedding_matrix = np.zeros((10000, 100)) # 10000 - num_words, 100 - length of sequence
for word, i in tokenizer.word_index.items():
    if i < 10000:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
import tensorflow as tf
from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Dense, Embedding, Flatten

model = tf.keras.models.Sequential([
    Embedding(10000, 100, input_length=100),
    Flatten(),
    Dense(32, activation=tf.nn.relu),
    Dense(1, activation=tf.nn.sigmoid),
])

model.layers[0].set_weights([embedding_matrix]) # Add glove to our model
model.layers[0].trainable = False # Freeze embedding layer

model.compile(
    loss = losses.binary_crossentropy,
    optimizer = optimizers.RMSprop(lr=0.001),
    metrics = ['accuracy'],
)

history = model.fit(
    train_data, train_labels,
    validation_data = [validation_data, validation_labels],
    batch_size = 32, 
    epochs = 10,
)
import matplotlib.pyplot as plt


plt.plot(history.history['accuracy'], 'r-', label='Train accuracy')
plt.plot(history.history['val_accuracy'], 'b-', label='Validation accuracy')
plt.legend()
plt.grid()
plt.show()

_, accuracy = model.evaluate(test_data, test_labels, verbose=0)
print('Test accuracy: ', round(accuracy * 100, 2), '%')
